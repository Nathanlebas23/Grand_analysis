import numpy as np
import matplotlib.pyplot as plt
from grand import Geodetic, GRANDCS


class DataQualityCheck :
    """
    This class is used to check the quality of data in a given dataset.
    """

    def __init__(self, data):
        """
        Initializes the DataQualityCheck class with the provided dataset.

        :param data: The dataset to be checked.
        """
        self.data = data

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
        


    def histo_DT_vs_time(self, bins=100):

        du_positions = self.get_du_pos() 
        t_ants_list = self.get_t_ants()     
        results = []
        for idx, (du, t_ants) in enumerate(zip(du_positions, t_ants_list)):
            