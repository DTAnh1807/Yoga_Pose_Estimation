from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
from demo import *
from pose_estimation import *

data_train = pd.read_csv("train_angle.csv")
data_test = pd.read_csv("test_angle.csv")


#data_train = data_train.drop(labels="Unnamed: 0", axis=1)  # delete if error about it
X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']


model = SVC(kernel='poly', decision_function_shape='ovo',probability=True)
model.fit(X, Y)
# mpPose = mp.solutions.pose
# pose = mpPose.Pose(static_image_mode=True, min_detection_confidence=0.2)
# mpDraw = mp.solutions.drawing_utils

# Test phase : build test dataset then evaluate
#data_test.drop(labels="Unnamed: 0", axis=1, inplace=True)  # delete if error about it
predictions = evaluate(data_test, model, show=True)

# Create a confusion matrix
# cm = confusion_matrix(data_test['target'], predictions)


#predict_video(model,'video_demo/yoga poses demo 2/yoga poses demo 2/goddess.mp4',show=True)
#predict('DATASET/TEST/goddess/00000092.png',model,show=True)
correct_feedback(model,'downdog_warrior2.mp4','teacher_yoga/angle_teacher_yoga.csv')

cv2.destroyAllWindows()
