from utils import detection_gt_extractor
from utils import annotation_parser
from paths import AICITY_DIR
from utils.object_tracking import Frame, ROI, ObjectTracker
import cv2
import glob
import os
import matplotlib.pyplot as plt
import motmetrics as mm


def load_detections_txt(detections_file, gtFormat, confidence_th=.2):
    detector = detection_gt_extractor.detectionExtractorGT(detections_file, gtFormat, confidence_th)
    # frame objects creation
    frame_dict = {i: Frame(i) for i in range(1,detector.getGTNFrames())}

    # filling frames with ROIs
    for i in frame_dict:
        for r in detector.getAllFrame(i):
            frame_dict[i].add_ROI( ROI(r[2], r[3], r[4], r[5]) )

    return frame_dict


def load_annotations(annotations_file, file_type='xml', gt_format='LTWH'):
    annotations = annotation_parser.annotationsParser(annotations_file, fileType=file_type, gtFormat=gt_format)

    # frame objects creation
    frame_dict = {i+1: Frame(i+1) for i in range(0, annotations.getGTNFrames())}

    # filling frames with ROIs
    for i in frame_dict:
        for r in annotations.getAllFrame(i):
            if r[-1] == 1:  # only if car
                frame_dict[i].add_ROI(ROI(r[2], r[3], r[4], r[5], r[1]))

    return frame_dict


def make_video_from_tracker(trckr, video_name, seqcam_path='S03/c010', plot=False):
    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), seqcam_path, 'frames/image-????.png')))
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, _ = cv2.imread(filepaths[0]).shape  # not all videos are fullHD, some are higher/lower resolution!
    video = cv2.VideoWriter(video_name, four_cc, 10, (width, height))
    for idx in range(1, len(filepaths)):
        print(filepaths[idx])  # comment this print to boost speed
        image = cv2.imread(filepaths[idx])
        image = trckr.draw_frame(idx, image)
        if plot:
            cv2.imshow("Image", image)
            cv2.waitKey(1)

        video.write(image)

    video.release()


def make_video_from_kalman_tracker(trckr, video_name, seqcam_path='S03/c010', plot=False):
    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), seqcam_path, 'frames/image-????.png')))
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, _ = cv2.imread(filepaths[0]).shape  # not all videos are fullHD, some are higher/lower resolution!
    video = cv2.VideoWriter(video_name, four_cc, 10, (width, height))
    for idx in range(1, len(filepaths)):
        print(filepaths[idx])
        image = cv2.imread(filepaths[idx])
        image = trckr.draw_frame_kalman(idx, image)
        if plot:
            cv2.imshow("Image", image)
            cv2.waitKey(1)

        video.write(image)

    video.release()


def print_mot_metrics(acc):
    # print final metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'],
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)
