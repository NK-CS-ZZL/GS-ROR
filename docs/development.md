# Development for evaluation and training



## Datasets Preparation<p id="Datasets"></p>

Before evaluation, you should:
+ Prepare standard datasets for evaluation and training:

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
    <th> :link: Checkpoint </th>
    <th> :link: Result </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Glossy Synthetic</td>
    <th> <a href="https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FNeRO&ga=1">Images</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
  </tr>
  <tr>
    <td>Shiny Blender</td>
    <th> <a href="https://storage.googleapis.com/gresearch/refraw360/ref.zip">Images</a> / <a href="https://drive.google.com/file/d/1HGTD3uQUr8WrzRYZBagrg75_rQJmAK6S/view?usp=sharing">Point Cloud</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
  </tr>
  <tr>
    <td>TensoIR Synthetic</td>
    <th> <a href="https://zenodo.org/records/7880113#.ZE68FHZBz18">Images</a> / <a href="https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view">Env. maps</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
    <th><a href="https://huggingface.co/lalala125/GSROR">HuggingFace</a></th>
  </tr>

</tbody>
</table>

**Update:** Now you can also download our checkpoints from [HuggingFace](https://huggingface.co/lalala125/DiscreteSDF).

+ Check the dataroot is organized as follows:

```shell
./data
├── glossy_relight
├── GlossySynthetic
│   ├── angel
│   ├── ...
│   └── teapot
├── GlossySynthetic_blender
│   ├── angel
│   ├── ...
│   └── teapot
├── shiny_blender 
│   ├── ball
│   ├── ...
│   └── toaster
├── tensoir_synthetic 
│   ├── armadillo
│   ├── env
│   ├── ficus
│   ├── hotdog
│   └── lego
└── ShinyBlender_eval
    ├── ball
    ├── ...
    └── toaster
```

2. Format the Glossy Synthetic Dataset

```shell
python nero2blender.py --path [PATH_TO_DATASET]
python nero2blender_relight.py --path [PATH_TO_DATASET]
```


3. Download the provided checkpoints.

Then, you can evaluate all our models. Besides, you can use the given rendered results directly, which provided in the [Table](#Downloads).



## Evaluation

### Glossy Synthetic Dataset

**NVS Quality** You can use the below script to evaluate the NVS quality in terms of PSNR, SSIM, and LPIPS. For better NVS quality, we follow the GaussianShader shading model and training/test split (96/32).

```shell
sh relit_scripts/eval_render_glossy.sh [OUTPUT_DIR]
``` 

**Relighting Quality**: You can use the below script to evaluate the relighting quality in terms of PSNR, SSIM, and LPIPS, and we rescale the relghting image following [NeRO](https://github.com/liuyuan-pal/NeRO).

```shell
sh relit_scripts/eval_relit_glossy.sh [OUTPUT_DIR]
sh relit_scripts/avg_relit_glossy.sh [OUTPUT_DIR]
``` 

**Mesh Quality**: We follow the evaluation script from [NeRO](https://github.com/liuyuan-pal/NeRO) to evaluate the mesh quality on the Glossy Synthetic Dataset. You can clone the NeRO repo and follow their installation instruction. After that, you can run the following script to evaluate the mesh qualty:

```shell
python eval_synthetic_shape.py --mesh [MESH_PATH] --object [OBJ_NAME] 
# e.g., bell, cat
```

### Shiny Blender Dataset

**NVS Quality** You can use the below script to evaluate the NVS quality in terms of PSNR, SSIM, and LPIPS. For better NVS quality, we follow the GaussianShader shading model and training/test split (96/32).

```shell
sh relit_scripts/eval_render_render.sh [OUTPUT_DIR]
``` 


**Normal Quality**: You can evaluate the normal quality in terms of MAE, using the below script:

```shell
python eval/metrics_mae.py -i/--img_paths [IMG_PATH] -g/--gt_paths [GT_PATH]
# e.g., 
# python eval/metrics_mae.py -i result/shiny/coffee/relight/bridge -g data/shiny_blender/coffee/test
```

### TensoIR Synthetic Dataset

**Relighting Quality**: You can use the below script to evaluate the relighting quality in terms of PSNR, SSIM, and LPIPS, and we rescale the albedo while relighting following [TensoIR](https://github.com/Haian-Jin/TensoIR).

```shell
sh relit_scripts/eval_relit_tir.sh [OUTPUT_DIR]
sh relit_scripts/avg_relit_tir.sh [OUTPUT_DIR]
``` 


## Train yourself model from the scratch


All the training scripts are in the folder `scripts/train_scripts`. For example, if you want to train a model using the Glossy synthetic dataset, you can run:

```shell
sh scripts/train_scripts/train_glossy.sh
``` 
* Note: You may need to adjust the bounding box for TensoSDF and ensure it to cover the whole object.

Then you'll obtain all relightable model in the folder `outputs/relight/glossy`. If you want to extract the mesh further, you can finetune the GS model and extract the meshes as

```shell
sh scripts/mesh_scripts/finetune_sdf.sh outputs/relight/glossy
``` 


