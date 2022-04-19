import numpy as np
import cv2 as cv
import yaml

from random import randint, random
from glob import glob
from shapely.geometry import Polygon as ShapelyPolygon
from os.path import exists
from os import mkdir


SCENE_WIDTH = 416
SCENE_HEIGHT = 416


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

    def __init__(self, suit: str, value: str, cardClass: int, image):
        self.suit = suit
        self.value = value
        self.name = f"{value}{suit}"
        self.cardClass = cardClass
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
        card = Card(self.suit, self.value, self.cardClass, croppedImage)
        card.extentsPolygon = extentsPolygon.transform(
            np.identity(3), -minX, -minY)
        card.cornerPolygons = list(
            a.transform(M, -minX, -minY) for a in self.cornerPolygons)

        return card

    def translated(self, offX: int, offY: int):
        card = Card(self.suit, self.value, self.cardClass, self.image.copy())

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
        self._dealIndex = 0

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

    def getRotatingNext(self):
        card = self.cardList[self._dealIndex]
        self._dealIndex = (self._dealIndex+1) % len(self.cardList)
        return card

    def writeLabelMap(self, path: str):
        # label_map.pbtxt
        with open(path, 'w') as writer:
            for card in self.cardList:
                writer.writelines([
                    "item {\n",
                    f"  id: {card.cardClass}\n",
                    f'  name: "{card.name}"\n',
                    "}\n\n",
                ])


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

    def getImage(self, *, drawVisiblePolygons: bool = False):
        image = self.image.copy()

        if drawVisiblePolygons:
            for cornerPolygons in self.visibleCardCorners.values():
                for cp in cornerPolygons:
                    cp.draw(image, (255, 255, 0), 1)

        return image


class YoloDatasetGenerator:

    def __init__(self, path: str, name: str):
        self.deck = Deck()
        self.backgroundImageSet = BackgroundImageSet()
        self.dsRoot = f"{path}/{name}"
        self._outCounter = 0

    def _buildAllPaths(self, pathList: list):
        for localPath in pathList:
            if not exists(localPath):
                mkdir(localPath)

    def _generateLabels(self,
                        indexFileName: str,
                        count: int,
                        *,
                        subFolder: str = None,
                        ):
        with open(f"{self.dsRoot}/{indexFileName}", "w") as listWriter:
            imagesDir = f"{self.dsRoot}/images"
            labelsDir = f"{self.dsRoot}/labels"
            if subFolder is not None:
                imagesDir = f"{imagesDir}/{subFolder}"
                labelsDir = f"{labelsDir}/{subFolder}"
                self._buildAllPaths([imagesDir, labelsDir])

            for _ in range(count):
                self._outCounter += 1
                imageName = f"scene_{self._outCounter:04}"
                imageFilePath = f"{imagesDir}/{imageName}.jpg"
                objectListFilePath = f"{labelsDir}/{imageName}.txt"
                print(f"Building scene {self._outCounter}...")
                cardCount = randint(1, 4)
                cardList = list(self.deck.getRotatingNext()
                                for _ in range(cardCount))
                scene = Scene(self.backgroundImageSet.getRandom(), cardList)
                cv.imwrite(imageFilePath, scene.getImage())

                height, width, _ = scene.image.shape
                with open(objectListFilePath, "w") as objectListWriter:
                    for card, cpList in scene.visibleCardCorners.items():
                        for cp in cpList:
                            minX, minY, maxX, maxY = cp.bounds
                            x = (minX + maxX) / 2 / width
                            y = (minY + maxY) / 2 / height
                            w = (maxX - minX + 1) / width
                            h = (maxY - minY + 1) / height
                            print(f"{card.cardClass} {x} {y} {w} {h}",
                                  file=objectListWriter)

                print(imageFilePath, file=listWriter)


class YoloV3DatasetGenerator(YoloDatasetGenerator):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

        self._buildAllPaths([
            path, self.dsRoot,
            f"{self.dsRoot}/config",
            f"{self.dsRoot}/images",
            f"{self.dsRoot}/labels",
            f"{self.dsRoot}/backup",
            f"{self.dsRoot}/checkpoints",
        ])

    def generateTrain(self, count: int):
        self._generateLabels('train.txt', count)

    def generateValid(self, count: int):
        self._generateLabels('val.txt', count)

    def finalize(self):
        with open(f"{self.dsRoot}/config/coco.data", "w") as writer:
            print(f"classes={len(self.deck.cardList)}", file=writer)
            print(f"train={self.dsRoot}/train.txt", file=writer)
            print(f"valid={self.dsRoot}/val.txt", file=writer)
            print(f"names={self.dsRoot}/config/coco.names", file=writer)
            print(f"backup={self.dsRoot}/backup", file=writer)

        with open(f"{self.dsRoot}/config/coco.names", "w") as writer:
            for card in self.deck.cardList:
                print(card.name, file=writer)


class YoloV5DatasetGenerator(YoloDatasetGenerator):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

        self._buildAllPaths([
            path, self.dsRoot,
            f"{self.dsRoot}/images",
            f"{self.dsRoot}/labels",
            f"{self.dsRoot}/checkpoints",
        ])

    def generateTrain(self, count: int):
        self._generateLabels('train.txt', count, subFolder='train')

    def generateValid(self, count: int):
        self._generateLabels('val.txt', count, subFolder='val')

    def finalize(self):
        data = dict()

        data['path'] = self.dsRoot
        data['train'] = 'images/train'
        data['val'] = 'images/val'
        data['test'] = None

        data['nc'] = len(self.deck.cardList)
        data['names'] = list(card.name for card in self.deck.cardList)

        with open(f"{self.dsRoot}/config.yaml", "w") as writer:
            yaml.dump(data, writer)


datasetGenerator = YoloV5DatasetGenerator(
    '/home/mbtowns/projects/oh-hell-scorekeeper/data/pytorch', 'v2')
datasetGenerator.generateTrain(52*10)
datasetGenerator.generateValid(52)
datasetGenerator.finalize()


# datasetGenerator = YoloDatasetGenerator(
#     '/Users/mbtowns/projects/oh-hell-scorekeeper/data/pytorch', 'v0-onecard')
# datasetGenerator.generateTrain(20)
# datasetGenerator.generateValid(4)
# datasetGenerator.finalize()


# card.display()
# cardList[0].rotatedScaled(30, 0.5).display()

# while True:
#     cardCount = randint(1, 4)
#     cardList = list(deck.getRandom() for _ in range(cardCount))
#     scene = Scene(backgroundImageSet.getRandom(), cardList)
#     cv.imshow('scene', scene.getImage())
#     cv.waitKey(0)

print('done')
