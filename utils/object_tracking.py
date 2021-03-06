import numpy as np
import cv2
import random
import copy
from utils.kalman_filter import KalmanFilter, KalmanFilter_ConstantVelocity, KalmanFilter_ConstantAcceleration
from utils.optical_flow_tracker import  OpticalFlowTracker
import motmetrics as mm
from paths import AICITY_DIR

class TrackedObject:
    # a object with its track
    # a track is a dictionary of frame_id and roi of the tracked
    #   object on the frame

    def __init__(self, id):
        self.objectId = id
        self.track = {}
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))

    def add_frame_roi(self, frame_id, roi):
        roi.objectId = self.objectId
        r = ROI(roi.xTopLeft, roi.yTopLeft, roi.xBottomRight, roi.yBottomRight, self.objectId)
        self.track[frame_id] = r

    def get_track(self):
        return self.track

    def drawObjectVideo(self):
        for frame_id, roi in self.track.items():
            if not roi.image is None:
                cv2.imshow("Image", roi.image)
                cv2.waitKey(1)

class ROI:
    # region of interest of a detected object
    # may or may not have an object id associated

    def __init__(self, xTopLeft, yTopLeft, xBottomRight, yBottomRight, objectId = -1):
        self.xTopLeft = xTopLeft
        self.yTopLeft = yTopLeft
        self.xBottomRight = xBottomRight
        self.yBottomRight = yBottomRight
        self.objectId = objectId

    def __str__(self):
        return '{} {} {} {}'.format(self.xTopLeft, self.yTopLeft, self.xBottomRight, self.yBottomRight)

    def overlap(self, otherROI):
        # computes overlap of two ROI (self and other)
        overlap = 0

        if self.xTopLeft > otherROI.xBottomRight or self.xBottomRight < otherROI.xTopLeft:
            return 0

        if self.yTopLeft > otherROI.yBottomRight or self.yBottomRight < otherROI.yTopLeft:
            return 0

        left = max(self.xTopLeft, otherROI.xTopLeft)
        right = min(self.xBottomRight, otherROI.xBottomRight)
        top = max(self.yTopLeft, otherROI.yTopLeft)
        bottom = min(self.yBottomRight, otherROI.yBottomRight)

        dx = abs(left - right)
        dy = abs(top - bottom)
        if (dx>=0) and (dy>=0):
            overlap = dx * dy

        return overlap

    def center(self):
        return [[(self.xTopLeft + self.xBottomRight) / 2], [(self.yBottomRight + self.yTopLeft) / 2]]

    def reposition(self, center):
        w = (self.xTopLeft - self.xBottomRight)
        h = (self.yBottomRight - self.yTopLeft)
        r = ROI(self.xTopLeft, self.yTopLeft, self.xBottomRight, self.yBottomRight, self.objectId)
        r.xTopLeft = center[0][0] - w / 2
        r.yTopLeft = center[1][0] - h / 2
        r.xBottomRight = center[0][0] + w / 2
        r.yBottomRight = center[1][0] + h / 2
        return r

    def cropImage(self, image):
        x = int(self.xTopLeft)
        y = int(self.yTopLeft)
        w = int(abs(self.xTopLeft - self.xBottomRight))
        h = int(abs(self.yBottomRight - self.yTopLeft))
        self.image = image[y:y+h, x:x+w].copy()

        #cv2.imshow("Image", self.image)
        #cv2.waitKey(1)




class KalmanTrackedObject(TrackedObject):
    # a object with its track
    # a track is a dictionary of frame_id and roi of the tracked
    #   object on the frame

    def __init__(self, id, initial_roi: ROI):
        self.objectId = id
        self.track = {}
        self.track_corrected = {}
        self.KF = KalmanFilter_ConstantAcceleration(initial_roi.center())  # KF instance to track this object
        #self.KF = KalmanFilter_ConstantVelocity(initial_roi.center())  # KF instance to track this object
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.estimated_velocity = False


    def add_frame_roi(self, frame_id, roi:ROI):
        roi.objectId = self.objectId
        r = ROI(roi.xTopLeft, roi.yTopLeft, roi.xBottomRight, roi.yBottomRight, self.objectId)
        self.track[frame_id] = r

        center = roi.center()
        if not self.estimated_velocity:
            self.KF.estimate_initial_velocity(center)
            self.estimated_velocity = True
        new_center = self.KF.predict()
        #self.KF.predict()
        #new_center = self.KF.correct(center, 1)
        self.KF.correct(center, 1)
        raux = roi.reposition(new_center)
        r_c = ROI(raux.xTopLeft, raux.yTopLeft, raux.xBottomRight, raux.yBottomRight, self.objectId)
        self.track_corrected[frame_id] = r_c

class OpticalFlowTrackedObject(TrackedObject):
    # a object with its track
    # a track is a dictionary of frame_id and roi of the tracked
    #   object on the frame

    def __init__(self, id, initial_roi: ROI, frame_id):
        self.objectId = id
        self.track = {}
        self.track_corrected = {}
        frame_path = AICITY_DIR.joinpath('frames',
                                         'image-{:04d}.png'.format(frame_id))
        initial_image = cv2.imread(str(frame_path))
        self.OF = OpticalFlowTracker(initial_image, initial_roi)
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.first_frame = True


    def add_frame_roi(self, frame_id, roi:ROI):

        if self.first_frame:
            roi.objectId = self.objectId
            r = ROI(roi.xTopLeft, roi.yTopLeft, roi.xBottomRight, roi.yBottomRight, self.objectId)
            self.track[frame_id] = r
            self.track_corrected[frame_id] = r
            self.first_frame = False
        else:
            roi.objectId = self.objectId
            r = ROI(roi.xTopLeft, roi.yTopLeft, roi.xBottomRight, roi.yBottomRight, self.objectId)
            self.track[frame_id] = r
            frame_path = AICITY_DIR.joinpath('frames',
                                             'image-{:04d}.png'.format(frame_id))
            image = cv2.imread(str(frame_path))
            new_center = self.OF.predict(image, self.objectId, frame_id, False)
            self.OF.correct(image, roi)
            raux = roi.reposition(new_center)
            r_c = ROI(raux.xTopLeft, raux.yTopLeft, raux.xBottomRight, raux.yBottomRight, self.objectId)
            self.track_corrected[frame_id] = r_c

class Frame:
    # collection of ROIs detected on an image frame

    def __init__(self, id):
        self.number = id
        self.ROIs = []

    def get_id(self):
        return  self.number

    def add_ROI(self, r):
        self.ROIs.append(r)

    def get_ROIs(self):
        return self.ROIs

    def remove_object(self, id):
        rois = [r for r in self.ROIs if r.objectId == id]
        for r in self.ROIs:
            if r.objectId == id:
                self.ROIs.remove(r)

class ObjectTracker:
    # class that tracks objects along a list of frames

    def __init__(self, method):
        self.method = method
        self.firstFrame = True
        self.trackedObjects = {}
        self.trackedFrames = {}
        self.lastObjectId = 0

    def load_annotated_frame(self, frame : Frame):
        tracked_frame = Frame(frame.get_id())

        # create a new TrackedObject for each ROI
        for r in frame.get_ROIs():

            if r.objectId in self.trackedObjects:
                obj = self.trackedObjects[r.objectId]
                obj.add_frame_roi(frame.get_id(), r)
            else:
                obj = TrackedObject(r.objectId)
                obj.add_frame_roi(frame.get_id(), r)
                self.trackedObjects[r.objectId] = obj

            tracked_frame.add_ROI( ROI(r.xTopLeft, r.yTopLeft, r.xBottomRight, r.yBottomRight, r.objectId) )

        #print("Adding new tracked frame {}".format(frame.get_id()))
        self.trackedFrames[frame.get_id()] = copy.copy(tracked_frame)

    def process_frame(self, frame : Frame):

        tracked_frame = None

        if self.firstFrame:
            tracked_frame = Frame(frame.get_id())

            # create a new TrackedObject for each ROI
            for r in frame.get_ROIs():
                id = self.lastObjectId + 1
                if self.method == "RegionOverlap":
                    obj = TrackedObject(id)
                elif self.method == 'OpticalFlow':
                    obj = OpticalFlowTrackedObject(id, r, frame.get_id())
                else:
                    obj = KalmanTrackedObject(id, r)
                obj.add_frame_roi(frame.get_id(), r)
                self.trackedObjects[id] = obj
                self.lastObjectId = id

                tracked_frame.add_ROI( obj.get_track()[frame.get_id()] )

            self.firstFrame = False

        else:
            if self.method == "RegionOverlap":
                tracked_frame = self._process_frame_overlap(frame)
            elif self.method == 'OpticalFlow':
                tracked_frame = self._process_frame_optical_flow(frame)
            else:
                tracked_frame = self._process_frame_kalman(frame)

            # update TrackedObjects with new discovered rois
            for roi in tracked_frame.get_ROIs():
                if roi.objectId == -1:
                    # new object found
                    id = self.lastObjectId + 1
                    if self.method == "RegionOverlap":
                        obj = TrackedObject(id)
                    elif self.method == 'OpticalFlow':
                        obj = OpticalFlowTrackedObject(id, roi, frame.get_id())
                    else:
                        obj = KalmanTrackedObject(id, roi)
                    obj.add_frame_roi(frame.get_id(), roi)
                    self.trackedObjects[id] = obj
                    self.lastObjectId = id

                else:
                    self.trackedObjects[roi.objectId].add_frame_roi(frame.get_id(), roi)


        #print("Adding new tracked frame {}".format(frame.get_id()))
        self.trackedFrames[frame.get_id()] = tracked_frame

    # gets a frame whose rois have no assigned object id and
    # returns a copy of the frame with assigned oject ids if
    # possible (-1 otherwise)
    def _process_frame_overlap(self, untracked_frame: Frame, overlap_th = 0.0, unique_objects = True):
        last_frame = self.trackedFrames[untracked_frame.get_id() - 1]

        # Create a tracked frame with current frame id
        tracked_frame = Frame(untracked_frame.get_id())

        last_frame_rois = last_frame.get_ROIs().copy()

        # for each untracked ROI in current frame
        for uROI in untracked_frame.get_ROIs():
            # calculate overlapping between current untracked roi
            # and all tracked rois from previous frame
            overlapping = np.asarray([uROI.overlap(tROI) for tROI in last_frame_rois])
            if len(overlapping) == 0:
                max_idx = -1
            else:
                max_idx = np.argmax(overlapping)

            if max_idx == -1:
                tObjId = -1

            elif np.max(overlapping) > overlap_th:

                best_tROI = last_frame_rois[max_idx]
                tObjId = best_tROI.objectId
                # print("Best match for {} is {}".format(uROI, best_tROI))

                if unique_objects:
                    # remove object from last_frame_rois to ensure that objects
                    # appear only once
                    last_frame_rois.pop(max_idx)

            else:
                tObjId = -1

            new_tROI = ROI(uROI.xTopLeft, uROI.yTopLeft, uROI.xBottomRight, uROI.yBottomRight, tObjId)
            tracked_frame.add_ROI(new_tROI)

        return tracked_frame

    # gets a frame whose rois have no assigned object id and
    # returns a copy of the frame with assigned oject ids if
    # possible (-1 otherwise)
    def _process_frame_kalman(self, untracked_frame: Frame, overlap_th = 0.0, unique_objects = True):
        last_frame = self.trackedFrames[untracked_frame.get_id() - 1]

        # Create a tracked frame with current frame id
        tracked_frame = Frame(untracked_frame.get_id())

        last_frame_rois = last_frame.get_ROIs().copy()

        # for each untracked ROI in current frame
        for uROI in untracked_frame.get_ROIs():
            # calculate overlapping between current untracked roi
            # and all tracked rois from previous frame
            overlapping = np.asarray([uROI.overlap(tROI) for tROI in last_frame_rois])
            if len(overlapping) == 0:
                max_idx = -1
            else:
                max_idx = np.argmax(overlapping)

            if max_idx == -1:
                tObjId = -1

            elif np.max(overlapping) > overlap_th:

                best_tROI = last_frame_rois[max_idx]
                tObjId = best_tROI.objectId
                # print("Best match for {} is {}".format(uROI, best_tROI))

                if unique_objects:
                    # remove object from last_frame_rois to ensure that objects
                    # appear only once
                    last_frame_rois.pop(max_idx)

            else:
                tObjId = -1

            new_tROI = ROI(uROI.xTopLeft, uROI.yTopLeft, uROI.xBottomRight, uROI.yBottomRight, tObjId)
            tracked_frame.add_ROI(new_tROI)

        return tracked_frame

    # gets a frame whose rois have no assigned object id and
    # returns a copy of the frame with assigned oject ids if
    # possible (-1 otherwise)
    def _process_frame_optical_flow(self, untracked_frame: Frame, overlap_th = 0.0, unique_objects = True):
        last_frame = self.trackedFrames[untracked_frame.get_id() - 1]

        # Create a tracked frame with current frame id
        tracked_frame = Frame(untracked_frame.get_id())

        last_frame_rois = last_frame.get_ROIs().copy()

        # for each untracked ROI in current frame
        for uROI in untracked_frame.get_ROIs():
            # calculate overlapping between current untracked roi
            # and all tracked rois from previous frame
            overlapping = np.asarray([uROI.overlap(tROI) for tROI in last_frame_rois])
            if len(overlapping) == 0:
                max_idx = -1
            else:
                max_idx = np.argmax(overlapping)

            if max_idx == -1:
                tObjId = -1

            elif np.max(overlapping) > overlap_th:

                best_tROI = last_frame_rois[max_idx]
                tObjId = best_tROI.objectId
                # print("Best match for {} is {}".format(uROI, best_tROI))

                if unique_objects:
                    # remove object from last_frame_rois to ensure that objects
                    # appear only once
                    last_frame_rois.pop(max_idx)

            else:
                tObjId = -1

            new_tROI = ROI(uROI.xTopLeft, uROI.yTopLeft, uROI.xBottomRight, uROI.yBottomRight, tObjId)
            tracked_frame.add_ROI(new_tROI)

        return tracked_frame

    def print_objects(self):
        for obj in self.trackedObjects:
            print('Object {}'.format(obj))
            for frame, roi in self.trackedObjects[obj].get_track().items():
                print('\tin frame {} at position {}'.format(frame, roi))

    def print_frames(self):
        for f in self.trackedFrames:
            print('Frame {}'.format(self.trackedFrames[f].get_id()))
            for roi in self.trackedFrames[f].get_ROIs():
                print('\tobject {} at position {}'.format(roi.objectId, roi))

    def draw_frame(self, frame_number, image):
        if image is None:
            return image

        overlay = image.copy()

        if frame_number in self.trackedFrames:
            for roi in self.trackedFrames[frame_number].get_ROIs():
                color = self.trackedObjects[roi.objectId].color

                cv2.rectangle(
                    overlay,
                    (int(roi.xTopLeft), int(roi.yTopLeft)),
                    (int(roi.xBottomRight), int(roi.yBottomRight)),
                    color,
                    -1
                )

                #roi_center = (int(roi.xTopLeft + (abs(roi.xTopLeft - roi.xBottomRight) / 2.0)),
                #              int(roi.yTopLeft + (abs(roi.yTopLeft - roi.yBottomRight) / 2.0)) )
                text_pos = (int(roi.xTopLeft+10), int(roi.yTopLeft+20))

                cv2.putText(image, str(roi.objectId), text_pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (0,0,0), 2)

            alpha = .7
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image


    def draw_frame_kalman(self, frame_number, image):
        frame = self.trackedFrames[frame_number]

        overlay = image.copy()
        for roi in frame.get_ROIs():
            color = self.trackedObjects[roi.objectId].color
            roiAux = self.trackedObjects[roi.objectId].track_corrected[frame_number]

            cv2.rectangle(
                overlay,
                (int(roiAux.xTopLeft), int(roiAux.yTopLeft)),
                (int(roiAux.xBottomRight), int(roiAux.yBottomRight)),
                color,
                -1
            )

            # For identified object tracks draw tracking line
            #for i in range(len(tracker.tracks)):
            lastr = None
            lastr_corrected = None
            corrected = True
            for i in range(frame_number):
                if not corrected:
                    if lastr == None and self.trackedObjects[roi.objectId].track.__contains__(i):
                        lastr = self.trackedObjects[roi.objectId].track[i]
                    elif self.trackedObjects[roi.objectId].track.__contains__(i):

                        # Draw trace line
                        r = self.trackedObjects[roi.objectId].track[i]
                        x1 = lastr.center()[0][0]
                        y1 = lastr.center()[1][0]
                        x2 = r.center()[0][0]
                        y2 = r.center()[1][0]
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                 color, 2)
                        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                                 color, 2)
                        lastr = r
                else:
                    if lastr_corrected == None and self.trackedObjects[roi.objectId].track_corrected.__contains__(i):
                       lastr_corrected = self.trackedObjects[roi.objectId].track_corrected[i]
                    elif self.trackedObjects[roi.objectId].track_corrected.__contains__(i):

                       # Draw trace line
                       r = self.trackedObjects[roi.objectId].track_corrected[i]
                       x1 = lastr_corrected.center()[0][0]
                       y1 = lastr_corrected.center()[1][0]
                       x2 = r.center()[0][0]
                       y2 = r.center()[1][0]
                       cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                 color, 2)
                       cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                                color, 2)
                       lastr_corrected = r

            #roi_center = (int(roi.xTopLeft + (abs(roi.xTopLeft - roi.xBottomRight) / 2.0)),
            #              int(roi.yTopLeft + (abs(roi.yTopLeft - roi.yBottomRight) / 2.0)) )
            text_pos = (int(roi.xTopLeft+10), int(roi.yTopLeft+20))

            cv2.putText(image, str(roi.objectId), text_pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (0,0,0), 2)

        alpha = .7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image


    def compute_mot_metrics(self, other):
        acc = mm.MOTAccumulator(auto_id=False) # we will provide frame ids

        for id, frame in self.trackedFrames.items():
            if self.method == "RegionOverlap":

                detObjects = [r.objectId for r in frame.get_ROIs()]
                detROIs = [[r.xTopLeft, r.yTopLeft, r.xBottomRight, r.yBottomRight] for r in frame.get_ROIs()]

            else: # Kalman
                detObjects = [r.objectId for r in frame.get_ROIs()]
                aux = [self.trackedObjects[id] for id in detObjects]
                detROIs = [[r.track_corrected[id].xTopLeft,
                            r.track_corrected[id].yTopLeft,
                            r.track_corrected[id].xBottomRight,
                            r.track_corrected[id].yBottomRight] for r in aux]

            if id in other.trackedFrames.keys():

                gtObjects = [r.objectId for r in other.trackedFrames[id].get_ROIs()]
                gtROIs = [[r.xTopLeft, r.yTopLeft, r.xBottomRight, r.yBottomRight] for r in
                          other.trackedFrames[id].get_ROIs()]
            else:
                gtObjects = []
                gtROIs = []


            dists = mm.distances.iou_matrix(gtROIs, detROIs, max_iou=0.5)
            acc.update(gtObjects, detObjects, dists, id)

            #print("Compute metrics for frame {}".format(id))
            #print(acc.mot_events.loc[id])

        return acc

    def update_mot_metrics(self, other, acc):
        for id, frame in self.trackedFrames.items():
            if self.method == "RegionOverlap":

                detObjects = [r.objectId for r in frame.get_ROIs()]
                detROIs = [[r.xTopLeft, r.yTopLeft, r.xBottomRight, r.yBottomRight] for r in frame.get_ROIs()]

            else: # Kalman
                detObjects = [r.objectId for r in frame.get_ROIs()]
                aux = [self.trackedObjects[id] for id in detObjects]
                detROIs = [[r.track_corrected[id].xTopLeft,
                            r.track_corrected[id].yTopLeft,
                            r.track_corrected[id].xBottomRight,
                            r.track_corrected[id].yBottomRight] for r in aux]

            if id in other.trackedFrames.keys():

                gtObjects = [r.objectId for r in other.trackedFrames[id].get_ROIs()]
                gtROIs = [[r.xTopLeft, r.yTopLeft, r.xBottomRight, r.yBottomRight] for r in
                          other.trackedFrames[id].get_ROIs()]
            else:
                gtObjects = []
                gtROIs = []


            dists = mm.distances.iou_matrix(gtROIs, detROIs, max_iou=0.5)
            acc.update(gtObjects, detObjects, dists)

            #print("Compute metrics for frame {}".format(id))
            #print(acc.mot_events.loc[id])



    def removeStaticObjects(self, dist_threshold_px=20):
        toDelete = []
        for obj in self.trackedObjects:
            track_frames = sorted(self.trackedObjects[obj].get_track().keys())
            fr = self.trackedObjects[obj].get_track()[track_frames[0]]
            lr = self.trackedObjects[obj].get_track()[track_frames[-1]]

            if (abs(fr.xTopLeft-lr.xTopLeft) < dist_threshold_px) and (abs(fr.yTopLeft - lr.yTopLeft) < dist_threshold_px) and (abs(fr.xBottomRight - lr.xBottomRight) < dist_threshold_px) and (abs(fr.yBottomRight - lr.yBottomRight) < dist_threshold_px):
                toDelete.append(obj)

        for id in toDelete:
            self.trackedObjects.pop(id)

            for f in self.trackedFrames:
                self.trackedFrames[f].remove_object(id)

    def mergeTrackedObjects(self, id1, id2):
        if not id1 in self.trackedObjects:
            return

        if not id2 in self.trackedObjects:
            return

        # merge tracked objects
        obj1 = self.trackedObjects[id1]
        obj2 = self.trackedObjects[id2]

        for frame_id, roi in obj2.get_track().items():
            roi.objectId = obj1.objectId
            obj1.add_frame_roi(frame_id, roi)

        del self.trackedObjects[id2]

        # replace ids in tracked frames
        for i, f in self.trackedFrames.items():
            for roi in f.get_ROIs():
                if roi.objectId == id2:
                    roi.objectId = id1

    def objectReId(self, oldId, newId):
        if not oldId in self.trackedObjects:
            return

        if newId in self.trackedObjects:
            return

        obj = self.trackedObjects[oldId]

        for frame_id, roi in obj.get_track():
            roi.objectId = newId

        for f in self.trackedFrames:
            for roi in f.get_ROIs():
                if roi.objectId == oldId:
                    roi.objectId = newId

    def getImagesForROIs(self, video_file):
        idx = 1
        cap = cv2.VideoCapture(video_file)
        ret = True
        while ret:
            ret, image = cap.read()
            #cv2.imshow("Image", image)
            #cv2.waitKey(1)
            if image is not None:
                if idx in self.trackedFrames:
                    for roi in self.trackedFrames[idx].get_ROIs():
                        roi.cropImage(image)

                    for id, obj in self.trackedObjects.items():
                        if idx in obj.get_track():
                            for i, r in obj.get_track().items():
                                r.cropImage(image)

            idx += 1
        cap.release()

    def mergeSimilarObjects(self):
        toMerge = []

        for id1, obj1 in self.trackedObjects.items():
            for id2, obj2 in self.trackedObjects.items():
                if obj1.objectId != obj2.objectId:
                    f1 = sorted(obj1.get_track().keys())[-1]
                    f2 = sorted(obj2.get_track().keys())[0]
                    i1 = obj1.get_track()[f1].image
                    i2 = obj2.get_track()[f2].image


                    hsv1 = cv2.cvtColor(i1, cv2.COLOR_BGR2HSV)
                    hsv2 = cv2.cvtColor(i2, cv2.COLOR_BGR2HSV)

                    h_bins = 50
                    s_bins = 60
                    histSize = [h_bins, s_bins]
                    # hue varies from 0 to 179, saturation from 0 to 255
                    h_ranges = [0, 180]
                    s_ranges = [0, 256]
                    ranges = h_ranges + s_ranges # concat lists
                    # Use the 0-th and 1-st channels
                    channels = [0, 1]

                    hist1 = cv2.calcHist([hsv1], channels, None, histSize, ranges, accumulate=False)
                    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist2 = cv2.calcHist([hsv2], channels, None, histSize, ranges, accumulate=False)
                    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
                    if d < 50:
                        toMerge.append((id1,id2))

        for id1, id2 in toMerge:
            self.mergeTrackedObjects(id1, id2)
