
from utils.object_tracking import ObjectTracker
import os
import time
from week3.task2 import load_annotations, load_detections_txt, print_mot_metrics, make_video_from_kalman_tracker, make_video_from_tracker
from paths import AICITY_DIR, PROJECT_ROOT


def optical_flow_tracking(path_to_gt_annotations, path_to_detections, file_type='txt', gt_format='LTWH',
                          method='RegionOverlap', cam_seq_str='S03_c010', conf_thr=.2, plot_frames=False):
    # Load detections
    #untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_yolo3.txt'), "LTWH")
    #untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt'), "LTWH")
    #untracked_frames = load_detections_txt(os.path.join('week3', 'det_mask_rcnn.txt'), "TLBR")
    #untracked_frames = load_detections_txt(os.path.join('week3', 'det_retinanet.txt'), "LTWH", .5)
    #untracked_frames = load_detections_txt(PROJECT_ROOT.joinpath('week3', 'det_retinanet.txt'), "LTWH", .5)
    untracked_frames = load_detections_txt(path_to_detections, gtFormat=gt_format, confidence_th=conf_thr)

    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

    tracker.print_objects()

    video_name = "{0}_{1}_{2}.avi".format("Tracking", method, cam_seq_str.replace('/', '_'))
    if method == 'Kalman':
        make_video_from_kalman_tracker(tracker, video_name, seqcam_path=cam_seq_str, plot=plot_frames)
    else:
        make_video_from_tracker(tracker, video_name, seqcam_path=cam_seq_str, plot=plot_frames)

    #Load annotations
    annotated_frames = load_annotations(path_to_gt_annotations, file_type=file_type, gt_format=gt_format)
    annotation_tracker = ObjectTracker("")

    for id, frame in annotated_frames.items():
        #print("Loading annotated gt frame {}".format(id))
        annotation_tracker.load_annotated_frame(frame)

    annotation_tracker.print_objects()
    annotation_tracker.print_frames()
    video_name = "{0}_{1}.avi".format("Annotations", cam_seq_str)
    make_video_from_tracker(annotation_tracker, video_name, seqcam_path=cam_seq_str, plot=plot_frames)

    acc = tracker.compute_mot_metrics(annotation_tracker)
    print_mot_metrics(acc)


if __name__ == "__main__":
    # Parameters
    method = "RegionOverlap"
    # method = "Kalman"
    # method = "OpticalFlow"
    fileType = 'txt'
    gtFormat = 'LTWH'
    detect_file = 'det_ssd512.txt'
    min_conf = .2
    vizFrames = False  # way quicker (we only write to disk)
    sequences_ids = [3]  # start "only" with all the cameras in sequence 3
    cam_ids = {1: range(1, 5+1), 3: range(11, 15+1), 4: range(16, 40+1)}  # ignore c010 (already computed)

    # Iterate for all combinations for a particular method (and sequence)
    for seq_id in sequences_ids:
        seq_cam_ids = cam_ids[seq_id]
        for cam in seq_cam_ids:  # for each camera, compute tracking
            print("Computing tracking for sequence 'S{0:02d}', camera ID 'c{1:03d}'...".format(seq_id, cam))
            start_timer = time.time()
            # Load ground-truth annotations
            folder_seq_cam = 'S{0:02d}/c{1:03d}'.format(seq_id, cam)
            path_to_gt_anns = PROJECT_ROOT.joinpath(AICITY_DIR, folder_seq_cam)
            if fileType == 'txt':  # format the complete path correctly by joining 'gt/gt.txt'
                path_to_gt_anns = os.path.join(path_to_gt_anns, 'gt/gt.txt')
            # for an xml, we expect the absolute path to the file
            path_to_detect = PROJECT_ROOT.joinpath(AICITY_DIR, folder_seq_cam, 'det', detect_file)
            optical_flow_tracking(path_to_gt_anns, path_to_detect, file_type=fileType, gt_format=gtFormat,
                                  method=method, cam_seq_str=folder_seq_cam, conf_thr=min_conf, plot_frames=vizFrames)
            end_timer = time.time()
            print("Computing tracks for this sequence took")
