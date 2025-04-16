#!/bin/sh

#SBATCH --job-name=MD_202250402_job    # Nom du job
#SBATCH --output=MD_test_%j.log   # Standard output et error log

#SBATCH --partition=htc               # Choix de partition (htc par défaut)

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=100000                   # Mémoire en MB par défaut
#SBATCH --time=0-06:00:00             # Délai max = 7 jours

#SBATCH --mail-user=nlebas@lpnhe.in2p3.fr          # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles




# init conda installed by grand experience
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh

# init GRANDLIB environment
conda activate /sps/grand/software/conda/grandlib_2304
echo beggin env

# init GRANDLIB 
cd /sps/grand/nlebas/grand/
source env/setup.sh
echo beggin main 

cd Grand_analysis/
python3 main.py
