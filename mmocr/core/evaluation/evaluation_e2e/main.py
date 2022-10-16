from text_evaluation import TextEvaluator

eval_dataset = 'ctw1500'
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
cfg['INFERENCE_TH_TEST'] = 0.4 # tune this parameter to achieve best result
e = TextEvaluator(dataset_name, cfg, False, output_dir=outdir)
res = e.evaluate()
print(res)
