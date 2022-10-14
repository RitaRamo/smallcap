from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import sys
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='coco-caption'
#dataType = sys.argv[1]
algName = 'fakecap'
annFile=sys.argv[1]#'%s/annotations/captions_%sKarpathy.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
resFile = sys.argv[2]

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

outfile = resFile.replace('preds', 'res')
outfile = outfile.replace('json', 'txt')

with open(outfile, 'w') as outfile:
  for metric, score in cocoEval.eval.items():
    outfile.write( '%s: %.2f\n'%(metric, score*100) )
