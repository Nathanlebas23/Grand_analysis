import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from iminuit import minimize
from PWF_reconstruction.recons_PWF import PWF_semianalytical
import scipy.optimize as so
from grand_analysis.wavefronts import SWF_loss  # SWF_loss must be defined in your module 'wavefronts'
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D projection
from grand import Geodetic, GRANDCS



class Reconstructor:
    """
    Class to perform event reconstruction using processed data.
    Assumes the input DataProcessor provides flat arrays for DU properties and a 
    separate 'mult' attribute with the multiplicity (number of triggered DUs) per event.
    """
    def __init__(self, data_processor, visualizer=None, rec_model='SWF', nb_min_antennas=5, event_index=None):
        """
        :param data_processor: Instance of DataProcessor with processed data.
        :param visualizer: (Optional) Instance of a Visualizer for coordinate conversion.
        :param event_index: (Optional) Index of a single event to analyze.
                            If None, the reconstruction is performed for all events.
        """
        self.data = data_processor
        self.event_index = event_index
        self.visualizer = visualizer
        self.nb_min_antennes = nb_min_antennas
        self.rec_model = rec_model
        
    def get_du_pos(self):
        """
        Converts DU geodetic coordinates (latitude, longitude, altitude) into Cartesian coordinates,
        grouping them per event.
        
        :return: List of numpy arrays; each array corresponds to one event with shape (N, 3).
        """
        # Define reference DAQ position as a Geodetic coordinate.
        daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)
        event_indices = self.get_event_indices()
        all_positions = []
        for start, end in event_indices:
            event_lat = self.data._du_lat[start:end]
            event_lon = self.data._du_long[start:end]
            event_alt = self.data._du_alt[start:end]
            event_positions = []
            for j in range(len(event_lat)):
                du_geo = Geodetic(
                    latitude=event_lat[j],
                    longitude=event_lon[j],
                    height=event_alt[j]
                )
                du_cartesian = GRANDCS(du_geo, obstime="2024-09-15", location=daq)
                event_positions.append([du_cartesian.x[0], du_cartesian.y[0], du_cartesian.z[0]])
            all_positions.append(np.array(event_positions))
        return all_positions

    def get_event_indices(self):
        """
        Computes start and end indices for each event from the flat data arrays,
        using the event multiplicities stored in the DataProcessor's mult array.
        
        :return: List of (start_index, end_index) tuples—one per event.
        """
        event_indices = []
        start = 0
        for m in self.data.mult:
            event_indices.append((start, start + m))
            start += m
        return event_indices

    def _get_coord_ants_single(self, index):
        """
        Retrieves Cartesian coordinates for all antennas (DUs) in a single event.
        This version uses array slices from the flat data arrays.
        
        :param index: Event index.
        :return: NumPy array of shape (number_of_antennas, 3) containing [x, y, z] coordinates.
        """
        event_indices = self.get_event_indices()
        if index >= len(event_indices):
            raise IndexError("Event index out of range.")
        start, end = event_indices[index]
        event_lat = self.data._du_lat[start:end]
        event_lon = self.data._du_long[start:end]
        event_alt = self.data._du_alt[start:end]
        daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=1215)
        
        event_geo = Geodetic(latitude=event_lat, longitude=event_lon, height=event_alt)
        event_geo_cs = GRANDCS(event_geo, obstime="2024-09-15", location=daq)
        coords = np.column_stack((event_geo_cs.x, event_geo_cs.y, event_geo_cs.z))
        return coords

    def _get_t_ants_single(self, index):
        """
        Retrieves and normalizes the trigger times for a single event.
        
        Normalization subtracts the mean trigger time.
        
        :param index: Event index.
        :return: NumPy array of normalized trigger times.
        """
        event_indices = self.get_event_indices()
        if index >= len(event_indices):
            raise IndexError("Event index out of range.")
        start, end = event_indices[index]
        true_times = self.data._trigger_secs - np.min(self.data._trigger_secs) + self.data._trigger_nanos * 1e-9
        event_times = np.array(true_times[start:end])
        event_times -= np.mean(event_times)
        return event_times

    def get_t_ants(self):
        """
        Retrieves sorted trigger times either for a single event or for all events.
        
        :return: If self.event_index is None, returns a list of sorted arrays (one per event).
                 Otherwise, returns a single sorted NumPy array.
        """
        if self.event_index is None:
            event_indices = self.get_event_indices()
            all_times = []
            for i in range(len(event_indices)):
                t = self._get_t_ants_single(i)
                all_times.append(t)
            return all_times
        else:
            t = self._get_t_ants_single(self.event_index)
            return np.sort(t)

    def reconstruct(self, event_index=None):
        """
        Reconstructs event directions using the semi-analytical PWF algorithm.
        
        Two operating modes:
          - If no event_index is provided (and self.event_index is also None),
            reconstruction is applied to all events and a list of reconstruction 
            dictionaries is returned.
          - If event_index (or self.event_index) is provided, only that event is reconstructed.
        
        Each reconstruction result (for a single event) is a dictionary containing:
            "event_index" - the event number,
            "coords"    - the Cartesian coordinates for the event,
            "t_ants"    - the sorted (and normalized) trigger times for the event,
            "theta"     - the reconstructed zenith angle (radians), and
            "phi"       - the reconstructed azimuth angle (radians).
        
        :param event_index: Optional single event index to reconstruct.
        :return: For a single event, a reconstruction dictionary. Otherwise, a list of dictionaries.
        """
        
        if self.rec_model== "PWF" :
            if event_index is None:
                if self.event_index is not None:
                    event_index = self.event_index
                else:
                    # Process all events.
                    du_positions = self.get_du_pos() 
                    t_ants_list = self.get_t_ants()     
                    results = []
                    for idx, (du, t_ants) in enumerate(zip(du_positions, t_ants_list)):
                        if len(t_ants) < self.nb_min_antennes:
                            continue
                        try:
                            theta, phi = PWF_semianalytical(du, t_ants)
                            print(f"Event {idx}: theta = {theta}, phi = {phi}")
                            if np.all(np.isnan(theta)) or (hasattr(phi, '__len__') and len(phi) == 0):
                                continue
                            results.append({
                                "event_index": idx,
                                "coords": du,
                                "t_ants": t_ants,
                                "theta": theta,
                                "phi": phi
                            })
                        except Exception as e:
                            print(f"Error processing event {idx}:", e)
                    return results
           
            else : 
                # Process a single event.
                event_indices = self.get_event_indices()
                if event_index > len(event_indices):
                    raise IndexError("Event index out of range.")
                # Get the full set of Cartesian coordinates and trigger times for the given event.
                du_positions = self._get_coord_ants_single(event_index)   # 2D array: shape (N, 3)
                t_ants = self._get_t_ants_single(event_index)               # 1D array: shape (N,)
                try:
                    # Pass the full array of coordinates and trigger times to the PWF function.
                    theta, phi = PWF_semianalytical(du_positions, t_ants)
                    print(f"Event {event_index}: theta = {theta}, phi = {phi}")
                    result = {
                        "event_index": event_index,
                        "coords": du_positions,
                        "t_ants": t_ants,
                        "theta": theta,
                        "phi": phi
                    }
                except Exception as e:
                    print(f"Error processing event {event_index}: {e}")
                    result = None
                return result


        elif self.rec_model == "SWF":
            if self.event_index is None:
                results = []
                event_indices = self.get_event_indices()
                for idx in range(len(event_indices)):   # ncoin
                    result = self._reconstruct_swf_single(idx)
                    if result is not None:
                        results.append(result)
                return results
            else:
                
                return self._reconstruct_swf_single(self.event_index)

        else:
            print(f"Reconstruction model '{self.rec_model}' not implemented.")
            return None

    def _reconstruct_pwf_single(self, event_index=None):
        event_indices = self.get_event_indices()
        if event_index > len(event_indices):
            raise IndexError("Event index out of range.")
        # Get the full set of Cartesian coordinates and trigger times for the given event.
        du_positions = self._get_coord_ants_single(event_index)   # 2D array: shape (N, 3)
        t_ants = self._get_t_ants_single(event_index)               # 1D array: shape (N,)
        try:
            # Pass the full array of coordinates and trigger times to the PWF function.
            theta, phi = PWF_semianalytical(du_positions, t_ants)
            print(f"Event {event_index}: theta = {theta}, phi = {phi}")
            result = {
                "event_index": event_index,
                "coords": du_positions,
                "t_ants": t_ants,
                "theta": theta,
                "phi": phi
            }
        except Exception as e:
            print(f"Error processing event {event_index}: {e}")
            result = None
        return result
    

    def _reconstruct_swf_single(self, event_index):
        """
        Reconstructs a single event using the spherical wavefront (SWF) model.
        Returns a dictionary with the reconstruction results or None if the reconstruction fails.
        """
        # Get event coordinates and trigger times
        coords = self._get_coord_ants_single(event_index)
        t_ants = self._get_t_ants_single(event_index)
        if len(t_ants) < self.nb_min_antennes:
            print(f"Not enough antennas for event {event_index}.")
            return None

        # Sort trigger times and reorder coordinates accordingly
        sort_idx = np.argsort(t_ants)
        coords_sorted = coords[sort_idx] # For Chi²
        t_sorted = np.sort(t_ants)       # For Chi²
        
        # Obtain an initial guess using your existing PWF reconstruction
        pwf_result = self._reconstruct_pwf_single(event_index=event_index)
        if pwf_result is None:
            print(f"PWF reconstruction failed for event {event_index}. Skipping SWF reconstruction.")
            return None
        
        theta_init = pwf_result["theta"]
        phi_init = pwf_result["phi"]

        

        # Define bounds for the SWF parameters [theta, phi, r_xmax, t_s].
        theta_deg = np.degrees(theta_init)
        phi_deg = np.degrees(phi_init)
        lower_theta = np.deg2rad(theta_deg - 5)
        upper_theta = np.deg2rad(theta_deg + 5)
        lower_phi = np.deg2rad(phi_deg - 15)
        upper_phi = np.deg2rad(phi_deg + 15)

        lb_r = -15.6e3 - 12.3e3/np.cos(np.deg2rad(theta_deg))
        ub_r = -6.1e3 - 15.4e3/np.cos(np.deg2rad(theta_deg))
        lb_ts = 6.1e3 + 15.4e3/np.cos(np.deg2rad(theta_deg))
        ub_ts = 0    


        # Print the bounds for debugging:
        # print(f"Event {event_index}: theta bounds: ({lower_theta}, {upper_theta}), "
        #     f"phi bounds: ({lower_phi}, {upper_phi}), r_xmax bounds: ({lb_r}, {ub_r}), "
        #     f"t_s bounds: ({lb_ts}, {ub_ts})")

        bounds = [[np.deg2rad(0),np.deg2rad(180)],
                                [np.deg2rad(0),np.deg2rad(360)], 
                               [0, 2000000],
                              [-2000000, 0]]
        

        initial_guess = np.array(bounds).mean(axis=1)
        
        try:
            method = 'migrad'
            res = minimize(SWF_loss, initial_guess, args=(coords_sorted, t_sorted, False),  
                            bounds=bounds, method=method)
        except Exception as e:
            print(f"Error in SWF minimization for event {event_index}: {e}")
            return None

        if not res.success:
            print(f"SWF minimization did not converge for event {event_index}: {res.message}")
            return None

        params_out = res.x
        chi2 = SWF_loss(params_out, coords_sorted, t_sorted, False)
        
        return {
            "event_index": event_index,
            "coords": coords_sorted,
            "t_ants": t_sorted,
            "theta": params_out[0],
            "phi": params_out[1],
            "r_xmax": params_out[2],
            "t_s": params_out[3],
            "chi2": chi2
        }


    def plot_3D_sphere(self):
        """
        Plots the reconstructed directional points on a 3D unit sphere.
        Theta and phi (in radians) are converted to Cartesian coordinates.
        """
        if self.event_index is None:
            results = self.reconstruct()
            if not results:
                print("No valid events for reconstruction.")
                return
            thetas = np.array([res["theta"] for res in results])
            phis = np.array([res["phi"] for res in results])
        else:
            res = self.reconstruct()
            if res is None:
                print("Reconstruction failed for the event.")
                return
            thetas = np.array([res["theta"]])
            phis = np.array([res["phi"]])
        xs = np.sin(thetas) * np.cos(phis)
        ys = np.sin(thetas) * np.sin(phis)
        zs = np.cos(thetas)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        sphere_x = np.outer(np.sin(v), np.cos(u))
        sphere_y = np.outer(np.sin(v), np.sin(u))
        sphere_z = np.outer(np.cos(v), np.ones_like(u))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='lightgray', alpha=0.3)
        ax.scatter(xs, ys, zs, color='red', s=10, alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Directional Distribution")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.show()

    def plot_2D_sphere(self):
        """
        Displays a 2D polar projection of the reconstructed directional points.
        Theta (zenith) angles (in degrees, clipped to 0°–90°) and phi (azimuth) angles are plotted.
        """
        if self.event_index is None:
            results = self.reconstruct()
            if not results:
                print("No valid events for 2D projection.")
                return
            thetas = np.array([res["theta"] for res in results])
            phis = np.array([res["phi"] for res in results])
        else:
            res = self.reconstruct()
            if res is None:
                print("Reconstruction failed for the event.")
                return
            thetas = np.array([res["theta"]])
            phis = np.array([res["phi"]])
        thetas_deg = np.clip(np.degrees(thetas), 0, 90)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.scatter(phis, thetas_deg, s=10, color='red', alpha=0.8)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_rlim(0, 90)
        tick_angles = np.arange(0, 360, 45)
        ax.set_xticks(np.radians(tick_angles))
        ax.set_xticklabels([f"{angle}°" for angle in tick_angles])
        ax.set_title("Azimuth (φ) and Zenith (θ)", va='bottom')
        plt.show()


    def plot_2D_sphere_chi2(self):
        """
        Displays a 2D polar projection of the reconstructed directional points
        with chi² values mapped to a color scale.
        The radius is based on the zenith angle (in degrees, clipped to [0, 100]) 
        and the color represents the chi² value.
        """
        if self.event_index is None:
            results = self.reconstruct()
            chi2_list = self.get_chi2()
            if not results:
                print("No valid events available for 2D chi² projection.")
                return
            thetas = np.array([res["theta"] for res in results])
            phis = np.array([res["phi"] for res in results])
            chis = np.array(chi2_list)
        else:
            res = self.reconstruct()
            if res is None:
                print("Reconstruction failed for the current event.")
                return
            chi2_list = self.get_chi2()
            try:
                chi2_value = chi2_list[self.event_index]
            except IndexError:
                print("Chi² value not found for the current event.")
                return
            thetas = np.array([res["theta"]])
            phis = np.array([res["phi"]])
            chis = np.array([chi2_value])
        
        # Convert theta to degrees and clip to [0, 100] for visualization
        thetas_deg = np.clip(np.degrees(thetas), 0, 100)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        scatter = ax.scatter(phis, thetas_deg, s=10, c=chis, cmap='viridis', alpha=0.8)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_rlim(0, 100)
        tick_angles = np.arange(0, 360, 45)
        ax.set_xticks(np.radians(tick_angles))
        ax.set_xticklabels([f"{angle}°" for angle in tick_angles])
        ax.set_title("Azimuth (φ), Zenith (θ) and Chi²", va='bottom')
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Chi²")
        plt.show()

    def get_chi2(self):
        """
        Computes the chi² statistic for each event by comparing the residuals between
        the expected and experimental trigger times.
        
        :return: List of chi² values (one per event).
        """
        chi2_list = []
        event_indices = self.get_event_indices()
        if not event_indices:
            print("No events available for chi² calculation.")
            return chi2_list

        # Compute the true trigger times (provided by the data processor)
        true_times = self.data.compute_true_time()
        
        for idx, (start, end) in enumerate(event_indices):
            du_coords = self._get_coord_ants_single(idx)
            if du_coords.size == 0:
                print(f"No DU coordinates available for event {idx}. Skipping event.")
                continue

            t_triggers = np.array(true_times[start:end])
            if len(t_triggers) < self.nb_min_antennes :
                continue

            # Sort trigger times and adjust DU coordinates accordingly
            sort_idx = np.argsort(t_triggers)
            t_triggers_sorted = t_triggers[sort_idx]
            du_coords_sorted = du_coords[sort_idx]

            # Reconstruct event to get theta and phi
            result = self.reconstruct(event_index=idx)
            if result is None:
                print(f"Reconstruction failed for event {idx}. Skipping event.")
                continue
            theta = result["theta"]
            phi = result["phi"]
            if np.isnan(theta) or np.isnan(phi):
                print(f"Invalid angles (NaN) for event {idx}. Skipping event.")
                continue

            # Calculate the shower axis vector (sinθ*cosφ, sinθ*sinφ, cosθ)
            shower_axis = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])

            # Calculate expected trigger times for each DU (in ns, speed of light c = 3e8 m/s)
            t_expected = du_coords_sorted.dot(shower_axis) / 3e8 * 1e9

            # Normalize experimental times so that the first trigger is zero
            t0 = t_triggers_sorted.min()
            t_exp = t_triggers_sorted - t0

            # Align reconstructed times relative to the first triggered DU
            index_min = np.argmin(t_triggers_sorted)
            time_offset = -t_expected[index_min]
            t_rec = t_expected + time_offset
            t_rec = -t_rec  

            
            residuals = t_exp - t_rec
            dof = len(t_exp) - 2
            chi2 = np.sum(residuals**2) / dof / 1e2 if dof > 0 else np.nan
            if chi2 > 1000 :
                continue
            chi2_list.append(chi2)
        
        return chi2_list

    def histo_thetaphi(self):
        """
        Plots a 2D histogram of the reconstructed theta (zenith) and phi (azimuth) angles
        for all events. Theta is converted to degrees for better interpretation.
        
        Note: This method should only be used when processing all events (event_index must be None).
        """
        
        results = self.reconstruct()  # Returns a list of reconstruction dictionaries.
        if not results:
            print("No valid events available for histogram.")
            return

        thetas = np.array([res["theta"] for res in results])
        phis = np.array([res["phi"] for res in results])

        # Convert theta from radians to degrees.
        thetas_deg = np.degrees(thetas)
        plt.figure(figsize=(8, 6))
        hist = plt.hist2d(phis, thetas_deg, bins=30, cmap='viridis')
        plt.xlabel("Azimuth (φ) [radians]")
        plt.ylabel("Zenith (θ) [degrees]")
        plt.title("2D Histogram of Reconstructed Theta and Phi")
        plt.colorbar(hist[3], label="Counts")
        plt.tight_layout()
        plt.show()


    def distrib_chi2_thetaphi(self):

        results = self.reconstruct()

        # Extraction des angles
        thetas = np.array([res["theta"] for res in results])
        phis = np.array([res["phi"] for res in results])
        thetas_deg = np.degrees(thetas)
        phis_deg = np.degrees(phis)
        
       
        # chi2_full = np.array(self.get_chi2())
        
        # log_chi2_full = np.log(chi2_full)

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogramme de χ²
        # ax[0, 0].hist(log_chi2_full, bins=100, color='blue', alpha=0.7)
        # ax[0, 0].set_xlabel("np.log(χ²)")
        # ax[0, 0].set_ylabel("Number")
        # ax[0, 0].set_title("log(χ²) distribution")
        
        # Histogramme des θ (en degrés)
        ax[0, 1].hist(thetas_deg, bins=100, color='green', alpha=0.7)
        ax[0, 1].set_xlabel("θ (degrees)")
        ax[0, 1].set_ylabel("Number")
        ax[0, 1].set_title("θ distribution")
        
        # Histogramme des φ (en radians)
        ax[1, 0].hist(phis_deg, bins=100, color='red', alpha=0.7)
        ax[1, 0].set_xlabel("φ (degrees)")
        ax[1, 0].set_ylabel("Number")
        ax[1, 0].set_title("φ distribution")
        
        # # Nuage de points θ vs φ coloré par chi²
        # sc = ax[1, 1].scatter(phis, thetas_deg, c=chi2_full, cmap='viridis', alpha=0.7)
        # ax[1, 1].set_xlabel("φ (radians)")
        # ax[1, 1].set_ylabel("θ (degrés)")
        # ax[1, 1].set_title("θ vs φ coloré par χ²")
        # fig.colorbar(sc, ax=ax[1, 1], label="χ²")
        
        plt.tight_layout()
        plt.show()

        return {"chi2": chi2_full, "theta": thetas, "phi": phis}
