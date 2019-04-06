import xmltodict


class annotationsParser():

    def __init__(self, filePath, fileType='xml', gtFormat='LTWH'):
        self.file = filePath
        self.gt = []
        self.nFrames = 0
        self.extractGT()
        self.fileType = fileType
        self.gtFormat = gtFormat

    def extractGT(self):  # for xml
        assert(self.fileType == 'xml')
        self.gt = []
        with open(self.file) as fd:
            doc = xmltodict.parse(fd.read())
        tracks = doc['annotations']['track']

        labelType = {
            1: 'car',
            2: 'bike',
            3: 'bicycle',
        }

        for track in tracks:

            BBoxes = track['box']
            id = track['@id']
            label = track['@label']
            if label == 'car':
                dataClass = 1
            elif label == 'bike':
                dataClass = 2
            elif label == 'bicycle':
                dataClass = 3
            else:
                dataClass = 0

            for BBox in BBoxes:
                # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
                if 'attribute' in BBox.keys():
                    #add_BBOX = BBox['attribute']['#text'] == 'false'
                    add_BBOX = True
                else:
                    add_BBOX = True
                if add_BBOX:
                    data = [int(BBox['@frame']),
                            int(id),
                            float(BBox['@xtl']),
                            float(BBox['@ytl']),
                            float(BBox['@xbr']),
                            float(BBox['@ybr']),
                            dataClass]
                    self.gt.append(data)

        for gtElement in self.gt:
            if int(gtElement[0]) > self.nFrames:
                self.nFrames = int(gtElement[0])
        return

    def extractGT_txt(self):  # for 'txt' gt annotations (aicity challenge format)
        assert(self.fileType == 'txt')
        self.gt = []
        with open(self.file, 'r') as fd:
            for line in fd:
                data = [float(elt.strip()) for elt in line.split(',')]
                if self.gtFormat == "LTWH":
                    # format data: [frame, ID, left, top, width, height, 1, -1, -1, -1]
                    data = [data[0], data[1], data[2], data[3], data[2] + data[4],
                            data[3] + data[5], data[6]]
                elif self.gtFormat == "TLBR":
                    # format data: [frame, ID, left, top, right, bottom, 1, -1, -1, -1]
                    data = [data[0], data[1], data[2], data[3], data[4],
                            data[5], data[6]]
                self.gt.append(data)

            for gtElement in self.gt:
                if int(gtElement[0]) > self.nFrames:
                    self.nFrames = int(gtElement[0])
            return

    def setFile(self, filePath):
        self.file = filePath

    def getGTFrame(self, i):
        gtElement = self.gt[i]
        return int(gtElement[0])

    def getGTNFrames(self):
        return self.nFrames

    def getGTID(self, i):
        gtElement = self.gt[i]
        return gtElement[1]

    def getGTBoundingBox(self, i):
        # BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [gtElement[2], gtElement[3], gtElement[4], gtElement[5]]
        return BBox

    def getGTList(self):
        return self.gt

    def getAllFrame(self, i):
        return [f for f in self.gt if f[0] == i]
