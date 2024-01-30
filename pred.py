from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Load the LFW dataset
lfw_people = fetch_lfw_people(data_home=lfw_home, min_faces_per_person=70, resize=0.4)