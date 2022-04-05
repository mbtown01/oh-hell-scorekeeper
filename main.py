import numpy as np
import cv2 as cv
import tensorflow as tf

from random import randint, random
from glob import glob
from shapely.geometry import Polygon as ShapelyPolygon
from object_detection.utils import dataset_util

SCENE_WIDTH = 300
SCENE_HEIGHT = 300


class Polygon(ShapelyPolygon):

    def draw(self, img, color: list, thickness: int):
        xCoords = list(int(a) for a in self.exterior.coords.xy[0])
        yCoords = list(int(a) for a in self.exterior.coords.xy[1])
        points = list(zip(xCoords, yCoords))
        for p1, p2 in zip(points[:-1], points[1:]):
            cv.line(img, p1, p2, color, thickness)

    def transform(self, matrix, shiftX=0, shiftY=0):
        xCoords, yCoords = self.exterior.coords.xy
        points = list(zip(xCoords, yCoords))[:-1]
        points = list(matrix @ (*a, 1) for a in points)
        points = list((int(x)+shiftX, int(y)+shiftY) for x, y, _ in points)
        return Polygon(points)


class Card:

    def __init__(self, suit: str, value: str, label: int, image):
        self.suit = suit
        self.value = value
        self.name = f"{value}{suit}"
        self.label = label
        self.image = image

        xMin, xMax = 8, 40
        yMin, yMax = 16, 91
        height, width, _ = self.image.shape

        self.extentsPolygon = self._buildRectanglePoly(0, 0, width, height)
        self.cornerPolygons = [
            self._buildRectanglePoly(
                xMin, yMin, xMax, yMax),
            self._buildRectanglePoly(
                width-xMax, height-yMax, width-xMin, height-yMin),
        ]

    def _buildRectanglePoly(self, xMin: int, yMin: int, xMax: int, yMax: int):
        points = [(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)]
        return Polygon(points)

    def rotatedScaled(self, angle: float, scale: float):
        assert scale <= 1.0
        height, width, _ = self.image.shape
        center = ((width-1)/2, (height-1)/2)
        M = cv.getRotationMatrix2D(center, angle, scale)
        newImage = cv.warpAffine(self.image, M, (width, height))

        M = np.concatenate([M, [[0, 0, 1]]], axis=0)
        extentsPolygon = self.extentsPolygon.transform(M)
        minX, minY, maxX, maxY = list(
            int(a) for a in extentsPolygon.bounds)

        croppedImage = newImage[minY:maxY+1, minX:maxX+1]
        card = Card(self.suit, self.value, self.label, croppedImage)
        card.extentsPolygon = extentsPolygon.transform(
            np.identity(3), -minX, -minY)
        card.cornerPolygons = list(
            a.transform(M, -minX, -minY) for a in self.cornerPolygons)

        return card

    def translated(self, offX: int, offY: int):
        card = Card(self.suit, self.value, self.label, self.image.copy())

        M = np.identity(3)
        card.extentsPolygon = self.extentsPolygon.transform(M, offX, offY)
        card.cornerPolygons = list(
            a.transform(M, offX, offY) for a in self.cornerPolygons)

        return card

    def drawOnSceneImage(self, scene, *, drawCornerPolygons: bool = True):
        minX, minY, maxX, maxY = list(
            int(a) for a in self.extentsPolygon.bounds)

        alpha = self.image[:, :, 3] / 255.0
        for color in range(3):
            background = \
                scene[minY:maxY+1, minX:maxX+1, color] * (1-alpha)
            change = self.image[:, :, color] * alpha
            scene[minY:maxY+1, minX:maxX+1, color] = \
                background.astype(np.uint8) + change.astype(np.uint8)

        if drawCornerPolygons:
            for polygon in self.cornerPolygons:
                polygon.draw(scene, (0, 0, 255), 1)
            self.extentsPolygon.draw(scene, (255, 0, 0), 1)


class Deck:

    def __init__(self):
        """ Loads the deck from the master PNG file """
        img = cv.imread('data/all-cards-cropped.png', cv.IMREAD_UNCHANGED)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rowBoundaries = Deck._findBoundaries(imgGray[:, 200])
        colBoundaries = Deck._findBoundaries(imgGray[20, :])
        suitLabels = ['C', 'H', 'S', 'D']
        valLabels = ['A', '2', '3', '4', '5', '6',
                     '7', '8', '9', 'T', 'J', 'Q', 'K']

        self.cardList = list()

        for (xMin, xMax), valLabel in zip(colBoundaries, valLabels):
            for (yMin, yMax), suitLabel in zip(rowBoundaries, suitLabels):
                image = img[yMin:yMax, xMin:xMax]
                self.cardList.append(
                    Card(suitLabel, valLabel, len(self.cardList), image))

    @classmethod
    def _findBoundaries(cls, range):
        range = np.array(list(1 if a > 0 else 0 for a in range))
        rangeDiff = range[1:] - range[:-1]
        rangeEdges = np.array(
            list(a for a, b in enumerate(rangeDiff) if b != 0))
        return list(zip(rangeEdges[::2], rangeEdges[1::2]))

    def getRandom(self):
        return self.cardList[randint(0, len(self.cardList)-1)]


class BackgroundImage:

    def __init__(self, fileName: str):
        self._fileName = fileName
        self._image = None

    def getImage(self):
        if self._image is None:
            self._image = cv.imread(self._fileName)
            self._image = cv.resize(
                self._image, (SCENE_HEIGHT, SCENE_WIDTH),
                interpolation=cv.INTER_CUBIC)
        return self._image


class BackgroundImageSet:

    def __init__(self):
        fileList = list(
            f for s in glob("../dtd/images/*") for f in glob(f"{s}/*.jpg"))
        print(f"Loaded {len(fileList)} images")
        self._imageList = list(BackgroundImage(f) for f in fileList)

    def getRandom(self):
        return self._imageList[randint(0, len(self._imageList)-1)]


class Scene:

    def __init__(self, backgroundImage: BackgroundImage, cardList: list):
        self.backgroundImage = backgroundImage
        self.cardList = list()
        self.image = backgroundImage.getImage().copy()

        finalHeight, finalWidth, _ = self.image.shape
        scale = random()*0.2 + 0.3

        for i, origCard in enumerate(cardList):
            while len(self.cardList) != i+1:
                angle = random()*180-90
                card = origCard.rotatedScaled(angle, scale)
                height, width, _ = card.image.shape
                offX = randint(0, finalWidth-width)
                offY = randint(0, finalHeight-height)
                card = card.translated(offX, offY)

                # Walk the cards that are down now in order from bottom up,
                # and make sure that every card has at least one corner
                # uncovered, otherwise randomly place the top card again
                cornerTestList = self.cardList + [card]
                if not any(
                    all(any(cp.intersects(e.extentsPolygon)
                            for e in cornerTestList[i+1:])
                        for cp in bottomCard.cornerPolygons)
                        for i, bottomCard in enumerate(cornerTestList)):
                    card.drawOnSceneImage(self.image, drawCornerPolygons=False)
                    self.cardList.append(card)

        self.visibleCardCorners = {
            card: [cp for cp in card.cornerPolygons
                   if not any(cp.intersects(e.extentsPolygon)
                              for e in self.cardList[i+1:])]
            for i, card in enumerate(self.cardList)
        }

    def buildTFExample(self):
        # Based on https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
        height, width, _ = self.image.shape
        # imageEncoded = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        imageEncoded = cv.imencode('.png', self.image)[1]
        imageEncoded = np.asarray(imageEncoded)
        imageEncoded = np.expand_dims(imageEncoded, axis=0)
        imageEncoded = bytes(imageEncoded)
        imageFormat = b'png'
        filename = b''  # Filename of the image. Empty if image is not from file

        xmins, xmaxs, ymins, ymaxs, classNames, classLabels = \
            [], [], [], [], [], [],
        for card, cornerPolygons in self.visibleCardCorners.items():
            for cp in cornerPolygons:
                minX, minY, maxX, maxY = cp.bounds
                xmins.append(minX / width)
                xmaxs.append(maxX / width)
                ymins.append(minY / height)
                ymaxs.append(maxY / height)
                classNames.append(card.name.encode('utf-8'))
                classLabels.append(card.label)

        return tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(imageEncoded),
            'image/format': dataset_util.bytes_feature(imageFormat),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classNames),
            'image/object/class/label': dataset_util.int64_list_feature(classLabels),
        }))

    def getImage(self, *, drawVisiblePolygons: bool = False):
        image = self.image.copy()

        if drawVisiblePolygons:
            for cornerPolygons in self.visibleCardCorners.values():
                for cp in cornerPolygons:
                    cp.draw(image, (255, 255, 0), 1)

        return image


deck = Deck()
backgroundImageSet = BackgroundImageSet()
# card.display()
# cardList[0].rotatedScaled(30, 0.5).display()

with tf.io.TFRecordWriter('scenes.tfrecord') as writer:
    for _ in range(10):
        cardCount = randint(1, 4)
        cardList = list(deck.getRandom() for _ in range(cardCount))
        scene = Scene(backgroundImageSet.getRandom(), cardList)
        record = scene.buildTFExample()
        writer.write(record.SerializeToString())


# while True:
#     cardCount = randint(1, 4)
#     cardList = list(deck.getRandom() for _ in range(cardCount))
#     scene = Scene(backgroundImageSet.getRandom(), cardList)
#     cv.imshow('scene', scene.getImage())
#     cv.waitKey(0)

print('done')


# Some guy wrote a tool to generate the TFRecords required to train our
# object detector below.  I think we can just DIRECTLY create TFRecords
# from the code above plus this example we can then use to train a network
# https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

# Also seems that if you have multiple objects in the same image you simply
# have two FULL TFRecords, each pointing to the same image

# looks like for testing purpuses, there's a reader that can look at what
# we generate
# https://github.com/sulc/tfrecord-viewer

# Looks like object_detection/g3doc/using_your_own_dataset.md has some docs
# on how to make your own tfrecord output
