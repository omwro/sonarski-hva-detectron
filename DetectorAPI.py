
import numpy as np
import tensorflow as tf
import time
import cv2

def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """


    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image

class Detector:
    def __init__(self, model_path, name=""):
        self.graph = tf.Graph()
        self.model_path = model_path
        self.model_name = name
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            self.graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                self.graph_def.ParseFromString(f.read())
                tf.import_graph_def(self.graph_def, name='')

    def detect_objects(self, img, threshold=0.3):
        """Runs the model and returns the object inside it
        Args:
        img (np_array)    -- input image
        threshold (float) -- threshold between (0,1)

        Returns:
        objects -- object list, each element is a dictionary that has [id, score, x1, y1, x2, y2] keys
        Ex: {'id': 16, 'score': 0.11703299731016159, 'x1': 42, 'y1': 6, 'x2': 55, 'y2': 27}
        """

        start_time = time.time()
        objects = []

        # start the session
        with tf.compat.v1.Session(graph=self.graph) as sess:

            # reshpae input image to give it to the network
            rows = img.shape[0]
            cols = img.shape[1]
            image_np_expanded = np.expand_dims(img, axis=0)

            # run the model
            (num, scores, boxes,
                classes) = self.sess.run(
                    [self.sess.graph.get_tensor_by_name('num_detections:0'),
                     self.sess.graph.get_tensor_by_name('detection_scores:0'),
                     self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                     self.sess.graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': image_np_expanded})

            # parse the results
            for i in range(int(num)):
                score = float(scores[0, i])
                if score > threshold:
                    obj = {}
                    obj["id"] = int(classes[0, i])
                    obj["score"] = score
                    bbox = [float(v) for v in boxes[0, i]]
                    obj["x1"] = int(bbox[1] * cols)
                    obj["y1"] = int(bbox[0] * rows)
                    obj["x2"] = int(bbox[3] * cols)
                    obj["y2"] = int(bbox[2] * rows)
                    objects.append(obj)

        end_time = time.time()

        return objects
