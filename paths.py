from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
# Only contained one camera of one sequence: S03 C010
# AICITY_DIR = PROJECT_ROOT.joinpath('data', 'AICity_data', 'train', 'S03',
#                                    'c010')
# Complete with all cameras (define only up to common base directory, then specify sequence and camera)
AICITY_DIR = PROJECT_ROOT.joinpath('data', 'AICity_all_train', 'aic19-track1-mtmc-train', 'train')
AICITY_ANNOTATIONS = PROJECT_ROOT.joinpath('full_annotations.xml')

DATA_DIR = PROJECT_ROOT.joinpath('data')
