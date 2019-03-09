import cv2

from evaluation import intersection_over
from utils import randomizer
from utils.detection_gt_extractor import detectionExtractorGT
from paths import AICITY_DIR


def run():
    intersection_over.test_iou()

    # Read GT from dataset
    gtExtractor = detectionExtractorGT(AICITY_DIR.joinpath('gt', 'gt.txt'))

    # parameters to randomize detections
    randomNoiseScale = 100
    additionDeletionProbability = 0.0

    TP = 0
    FN = 0
    FP = 0
    threshold = 0.5

    for i in range(gtExtractor.getGTNFrames()):

        # Get GT BBOX
        gt = []
        for j in range(len(gtExtractor.gt)):
            if gtExtractor.getGTFrame(j) == i:
                gtBBOX = gtExtractor.getGTBoundingBox(j)
                gt.append(gtBBOX)

        # Get detection BBOX
        detections = randomizer.randomize_detections(
            additionDeletionProbability,
            randomNoiseScale,
            gt[:]
        )
        BBoxesDetected = []

        for x in range(len(gt)):
            gtBBOX = gt[x]
            detection = []
            maxIoU = 0
            BBoxDetected = -1

            for y in range(len(detections)):
                iou = intersection_over.bb_intersection_over_union(gtBBOX, detections[y])
                if iou >= maxIoU:
                    maxIoU = iou
                    detection = detections[y]
                    BBoxDetected = y

            if maxIoU > threshold:
                TP = TP + 1
                BBoxesDetected.append(BBoxDetected)
            else:
                FN = FN + 1

            # load the image
            frame_path = AICITY_DIR.joinpath('frames',
                                             'image-{:04d}.png'.format(i + 1))
            image = cv2.imread(str(frame_path))

            # draw the ground-truth bounding box along with the predicted
            # bounding box
            cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 0, 255),
                2
            )
            cv2.rectangle(
                image,
                (int(gtBBOX[0]), int(gtBBOX[1])),
                (int(gtBBOX[2]), int(gtBBOX[3])),
                (0, 255, 0),
                2
            )

            # compute the intersection over union and display it

            cv2.putText(
                image,
                "IoU: {:.4f}".format(maxIoU),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            print("{}: {:.4f}".format(frame_path, maxIoU))

            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)

        for y in range(len(detections)):
            if not BBoxesDetected.__contains__(y):
                FP = FP + 1


if __name__ == '__main__':
    run()
