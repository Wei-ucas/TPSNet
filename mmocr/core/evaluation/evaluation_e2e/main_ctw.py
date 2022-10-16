import sys

from tools.evaluation_det_e2e_offline_v3.text_evaluation import TextEvaluator


import pickle
import numpy as np
import json
from mmdet.datasets.coco import COCO

def main(res_pkl):
    # coco_annos = COCO('../../data/totaltext/instances_test.json')
    coco_annos = COCO('/data/ww/mmocr/data/ctw1500_words/instances_test.json')
    results = pickle.load(open(res_pkl, 'rb'))
    img_ids = coco_annos.get_img_ids()
    output = []
    for i, img_id in enumerate(img_ids):
        img_name = coco_annos.load_imgs([img_id])[0]['file_name']
        # idx = str(img_id).zfill(7)
        res = results[i]
        ins = res['boundary_result']
        recs = res['strs']
        for j in range(len(ins)):
            pts = np.array(ins[j][:-1]).reshape(-1,2).tolist()
            score = ins[j][-1]
            rec = recs[j]
            out = {
                "image_id": int(img_name[5:-4]),
                # "image_id": i,
                "category_id": 1,
                "polys": pts,
                "rec": rec,
                "score": score
            }
            output.append(out)


    with open('ctw1500_res/my_text_results.json', 'w') as f:
        json.dump(output, f)





    eval_dataset = 'ctw1500'
    # eval_dataset = 'totaltext'
    if eval_dataset == 'ctw1500':
        dataset_name = ['ctw1500']
        outdir = 'ctw1500_res'
    elif eval_dataset == 'totaltext':
        dataset_name = ['totaltext']
        outdir = 'totaltext_res'
    elif eval_dataset == 'custom':
        dataset_name = ['custom']
        outdir = 'custom_res'
    cfg = {}
    for t in [0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
        print(t)
        cfg['INFERENCE_TH_TEST'] = t # tune this parameter to achieve best result
        e = TextEvaluator(dataset_name, cfg, False, output_dir= outdir)
        res = e.evaluate()
        print(res)

if __name__ == '__main__':
    res = sys.argv[1]
    main(res)