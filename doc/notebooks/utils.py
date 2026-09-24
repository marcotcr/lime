import os
import cv2
import json
import glob
import skimage
import requests
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from datetime import date,datetime
from IPython.display import display, HTML
pd.set_option('display.max_columns', None)
from skimage.segmentation import mark_boundaries
from matplotlib.colors import LinearSegmentedColormap

# from lime_stratified.lime.wrappers.scikit_image import SegmentationAlgorithm
from lime.wrappers.scikit_image import SegmentationAlgorithm
display(HTML("<style>.container { width:98% !important; }</style>"))
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
model = ResNet50(weights='imagenet')

#####################################################################################################################
#######################################         BASIC FUNCTIONS         #############################################
#####################################################################################################################
def check_folders(path_):
    ''' Args:
    path_: Path Verifier    '''
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f'folder created:\t{path_}')
def axis_off(ax):
    ax.set_xticks([], []) ; ax.set_yticks([], [])
#####################################################################################################################
#####################################    BLACKBOX MODEL PREDICTION FUNCTION #########################################
#####################################################################################################################
def get_ImageNet_ClassLabels(json_file=False):
    ''' Input:
            json_file:  Path to JSON File, if file is already downloaded, 
                        filepath can be passed as the input parameter
            json_file:  if file is not available locally then setting False will download the file automatically.  
        Output:
            class_names: A list containing the 1000 classes names from ImageNet dataset '''
    if os.path.isfile(json_file) == False:
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        r = requests.get(url, allow_redirects=True)
        open(json_file, 'wb').write(r.content)
        print('ImageNet Classes JSON File Downloaded')
    else:
        url = json_file
#         print('JSON File Loaded')
    with open(url) as file:   
        class_names = [v[1] for v in json.load(file).values()]
    return class_names

def load_model(model_name=None):
    if model_name=='ResNet50':
        model = ResNet50(weights='imagenet')
    if model:
        print('BlackBox Model Selected: \t\t',model_name)
        print('BlackBox Model Layers Count: \t\t',len(model.layers))
        print('BlackBox Model Weights Count: \t\t',len(model.weights))
        return model

def read_process_image(filename,model):
    ''' Args:
           dataset_name:    Name of Dataset
           filename:        Path to Image           '''
    rows,cols = model.input.shape[1],model.input.shape[2]
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,[rows,cols])
    return img
def get_class_idx_label_score(predicted,class_names):
    ''' 
    Args:
        predicted:  Prediction returned by Blackbox model 1x1000
        model_name: Model Name to be used to make predictions
        class_names:    List of class names for the dataset
    Result:
        PDI:                Predicted Class Index
        class_probability:  class_probability of current prediction
        PDL:                Predcited Class Label
        '''
    PDI = np.argmax(predicted)
    class_probability = predicted[0][PDI]
    PDL = class_names[PDI]
    return (PDI,class_probability,PDL)
def plot_save_prediction(X_s,PDL,class_probability,result_folder,file_name,save_image=False,plot_everything=False):
    '''
    Args:   
    X_s:    Image(3d numpy array)
    PDL:    Predicted Class Label
    class_probability:  Class Probablity or Predicirton Score returned by Blackbox Model
    curr_results:       Path to save the figure
    save_image:         Plot only if False, Plot and Save if True
    Result: '''
    plt.figure(figsize=(3,3))
    plt.imshow(X_s)
    if plot_everything:
        plt.title(str(PDL)+':'+str(class_probability))
    # plt.title(r'$\alpha > \beta$')
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if save_image:
        plt.savefig(result_folder+'//Predicted_'+str(PDL)+'_'+str(round(class_probability,3))+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()

#####################################################################################################################
#######################################    SEGMENTATION FUNCTIONS      ##############################################
#####################################################################################################################
# Create Segments 0-100, 100-200, 200-300
def segmentation_module(compute_segments,files,DS_path,sub_results_,model,segs_list,seg_algo):
    '''
    this function will require following parameters and will return a dataframe from csv file
    Args:    
        compute_segments: Boolean (True or False), if true, then it will compute the hyperparameters according to segments required
        files: a range of filename variable, upto which the segmentation parameters are required
        DS_path: Path to save or to load the segmentation parameters csv file
        model: Blackbox model loaded as variable
        model_name blackbox model name as string, to be 
        segs_list
    Result:
        df_seg: A Dataframe from CSV File either created after segmentation hyperparameters or from loaded file
    '''
    if compute_segments:     
        now = datetime.now()
        print("Segmentation Started\t\t:\t\t\t", now.strftime("%d/%m/%Y %H:%M:%S"))
        print('+'*94)
        print('| FileName\t | \tTarget Segments | Generated Segments | \tMax Distance | \tKernel Size  |')
        data_to_csv = dict()
        segs_param_table_sucess = []
        for f in files:
            print('-'*94)
            file_name = f'{f+1:08}'
            file = os.path.join(DS_path,'ILSVRC2012_test_'+file_name+'.JPEG')     
            image   = read_process_image(file,model)
            for srl in segs_list:
                target_seg_no = (srl if isinstance(srl, int) else srl[-1])
                md,ks,random_seed,ratio = search_segment_number(image, target_seg_no=target_seg_no, init_max_dist=100,
                                                                init_kernel_size=4,seg_algo=seg_algo)
                segments,segs,segmenter_fn = own_seg(image,md=md,ks=ks,random_seed=random_seed,ratio=ratio)
                segs_param_table = {'filename':file_name,'seg_algo':seg_algo, 'max_distance':md,
                                    'kernal_size':ks,'random_seed':random_seed,
                                    'ratio':ratio,'segments':segs,'target_segs':srl}
                segs_param_table_sucess.append(segs_param_table)
                print(f'| {file_name}\t | \t   {srl}\t\t| \t{segs}\t     | \t  {md:0.4}\t     |       {ks}\t     |')
                df_seg = pd.DataFrame(segs_param_table_sucess)
                df_seg.to_csv(f'{sub_results_}//Segmentation_Table_{segs_list}.csv', sep = ';' , index=False)
        now = datetime.now()
        print("Segmentation Completed\t\t:\t\t\t", now.strftime("%d/%m/%Y %H:%M:%S"))
    else:
        df_seg = pd.read_csv(f'{DS_path}//Segmentation_Table_{segs_list}.csv', sep = ';')
    return df_seg
######################################################################################################
############################   SUPPORTED FUNCTIONS FOR SEGMENTATION    ###############################
######################################################################################################
def get_segment_number(image, md,ks,seg_algo,random_seed=1234,ratio=0.2):
    ''' COMPUTE NO OF SEGMENTS by using hyperparametrs
    Args:
        image:       Image to be segmented (MxNx3) 
        md:          Max Distance
        ks:          Kernel Size
        seg_algo:    Algorithm used for segmentation '''
    segmentation_fn = SegmentationAlgorithm(seg_algo, kernel_size=ks, max_dist=md, ratio=ratio, random_seed=random_seed) 
    segments = segmentation_fn(image)
    return len(np.unique(segments))
def search_segment_number(image, target_seg_no, init_max_dist=100,init_kernel_size=4,seg_algo='quickshift'):
    ''' search_segment_number by implementing dichotomic_search
    Args:
    image:                Image to be segmented (MxNx3) 
    target_seg_no:        Target Segments Number       
    init_max_dist:        Initial Max Distance used for creating segments     
    init_kernel_size:     Initial Kernel Size used for creating segments    
    seg_algo:             Algorithm used for segmentation          
    Return:
    rmd:Max Distance Required to create Target Segments Number
    init_kernel_size:     Kernel Size Required to create Target Segments Number 
    '''
    random_seed=1234
    ratio=0.2
    lmd, rmd,ks = 0, init_max_dist,init_kernel_size
    lsn = get_segment_number(image, lmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
    rsn = get_segment_number(image, rmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
    niter = 0
    while niter<20 and rsn!=target_seg_no:
        niter += 1
        mmd = (lmd + rmd) / 2.0
        msn = get_segment_number(image, mmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
#         print(f'{lmd}:{lsn}  {mmd}:{msn} {rmd}:{rsn}')
        if msn <= target_seg_no <= lsn:
            rsn, rmd = msn, mmd
        else:
            lsn, lmd = msn, mmd
    return float(rmd),init_kernel_size,random_seed,ratio
            
def own_seg(image,md,ks,random_seed=1234,ratio=0.2,seg_algo='quickshift'):
    '''
    skimage.segmentation.quickshift(image, ratio=1.0, kernel_size=5, max_dist=10, 
    return_tree=False, sigma=0, convert2lab=True, rng=42, *, channel_axis=-1)
    '''
    ''' Function to Get Perform Segmentation using Quickshift and slic algorithms.
    Args:       
            X_s:        3d nupmy array 
            ks:         kernal  Size,float (Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters)
            md:         Max Dist, float, Cut-off point for data distances. Higher means fewer clusters
            seg_algo:   Segmentation Algorithm Name, Default: quickshift
            ratio:      float, optional, between 0 and 1
            sigma:      Width for Gaussian smoothing as preprocessing. Zero means no smoothing
    Results: 
            Segments:               Segments Created
            Segs:                   Number of Segments
            fn_segmentation:        Segmenter Function to take iamges and create segments'''
    if seg_algo == 'quickshift':
        segmenter_fn = SegmentationAlgorithm('quickshift', kernel_size=ks,max_dist=md, ratio=ratio,random_seed=random_seed)
    elif seg_algo == 'slic':
        segmenter_fn = SegmentationAlgorithm('slic',compactness=md,max_num_iter=ks, ratio=ratio,random_seed=random_seed)
    segments = segmenter_fn(image)
    segs = np.unique(segments).shape[0]
    def fn_segmentation(image):
            return segments
    return segments,segs,fn_segmentation

#####################################################################################################################
#############################################    PLOT FUNCTIONS    ##################################################
#####################################################################################################################
def plot_seg_image(image,segments,md,ks,sub_results,file_name,save_image=False,plot_everything=True,hide_x_y_ticks=True):
    '''
    Args:  
        immgg:  Input Image (3d numpy array)
        segs:   Number of segments
        md:     Max Distance (Segmentation Parameters)
        ks:     Kernel Size  (Segmentation Parameters)
        PDL:    Predicted Class Label
        class_probability:  Class Probablity or Predicirton Score returned by Blackbox Model
        file_name:          Path to save the figure
        save_image:         Plot only if False, Plot and Save if True

    Result:     
    '''
    segs = np.unique(segments).shape[0]
    immgg=skimage.segmentation.mark_boundaries(image, segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
    plt.figure(figsize=(3,3))
    plt.imshow(immgg)
    if plot_everything:
        plt.title(str(segs)+ '_'+str(md)+'_'+str(ks))
    if hide_x_y_ticks:
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//Segs_'+str(segs)+ '_'+str(md)+'_'+str(ks)+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
def plot_classification_score_examples(explanation,data,labels,class_probability,sub_results,ttl,draw_quantile=False,quantile=[0.05,0.95],save_image=False,plot_points=1000,plot_everything=True,hide_x_y_ticks=True):
    ''' Args:
            Explanation:        Explaination returned by Lime-Image
            data:               Data returned by LIME-Image (dense num_samples * num_superpixels)       
            labels:             Prediction Probabilities Matrix generated by LIME-Image          
            class_probability:  Class Probability or Prediction Score Returned by BlackBox Model 
            curr_results:       Path to Save the Figure 
            filenameee:         Filename to save the corrosponding Figure with into Relavant Directory      
            draw_quantile:      False, set it to True if quantile plotting on classification score is also needed 
            quantile:           Quantile Upper and Lower bound for classification score on Default [0.05-0.95]'''
    colors = ['#6d9eeb','#f9cb9c']
    cm = LinearSegmentedColormap.from_list("Custom", colors)
    x = [np.sum(d) / len(d) for d in data]
    TL = explanation.top_labels[0]
    y =labels[:,TL]
    segs = data.shape[1]
    nos = data.shape[0]
    plt.figure(figsize=(3,3))
    plt.scatter(x[:plot_points],y[:plot_points] , c =y[:plot_points] , cmap = cm, s=20 , lw = 0.5 , edgecolors = 'black')
    plt.scatter(x[0],y[0] , c='m' ,marker='x', s =200)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axhline(class_probability, ls = '--' , lw = 2 , color ='g' )
    if draw_quantile:
        q_lower = np.quantile(y,quantile[0])
        q_upper = np.quantile(y,quantile[1])
        plt.axhline(q_lower, ls = '--' , lw = 1 , color ='red' )
        plt.axhline(q_upper, ls = '--' , lw = 1 , color ='blue' )
    if plot_everything:
        plt.text(+0.02,y[0]-0.10, '$y=f(\\xi)$' , fontsize='15')
        plt.ylabel('$f(\\xi_x)$',fontsize='15')
        plt.xlabel('$|x|$' , fontsize='15')
    if hide_x_y_ticks:
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//ClassScore_'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
    
def plot_heatmap_lime(heatmap,maxval,sub_results,ttl,save_result=False,show_color_bar=False,color_bar_position='right',hide_x_y_ticks=True):
    plt.figure(figsize=(3,3))
    plt.imshow(heatmap , cmap='bwr', vmin = -maxval, vmax = maxval )#, cmap='cool')
    if hide_x_y_ticks:
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if show_color_bar==True:
        if color_bar_position == 'right':
            plt.colorbar(im,fraction=0.046, pad=0.04, orientation='horizontal')
        if color_bar_position == 'bottom':
            plt.colorbar(im,fraction=0.046, pad=0.04, orientation='vertical')
    if save_result:
        plt.savefig(sub_results+'//Heatmap_'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
def heatmap_from_beta(segments=None, beta=None):
    if segments is not None and beta is not None:
        heatmap = np.zeros_like(segments, dtype=np.float32)
        for segm, importance in enumerate(beta):
            heatmap[ segments==segm ] += importance 
        return heatmap
def plot_classification_score(ax,explanation,X,Y,class_probability,draw_quantile=False,
                              quantile=[0.05,0.95],save_image=False,plot_points=1000,plot_everything=True):
    colors = ['#6d9eeb','#f9cb9c']
    cm = LinearSegmentedColormap.from_list("Custom", colors)
    x = [np.sum(d) / len(d) for d in X]
    TL = explanation.top_labels[0]
    segs = X.shape[1]
    nos = X.shape[0]
    ax.scatter(x[:plot_points], Y[:plot_points] , c = Y[:plot_points] , cmap = cm, s=20 , lw = 0.5 , edgecolors = 'black')
    ax.scatter(x[0],Y[0] , c='m' ,marker='x', s =200)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.axhline(class_probability, ls = '--' , lw = 2 , color ='g' )
    if draw_quantile:
        q_lower = np.quantile(y,quantile[0])
        q_upper = np.quantile(y,quantile[1])
        ax.axhline(q_lower, ls = '--' , lw = 1 , color ='red' )
        ax.axhline(q_upper, ls = '--' , lw = 1 , color ='blue' )
    if plot_everything:
        ax.text(+0.02, Y[0]-0.10, '$y=f(\\xi)$' , fontsize='15')
        ax.set_ylabel('$f(\\xi_x)$',fontsize='15')
        ax.set_xlabel('$|x|$' , fontsize='15')
def get_img_mask_lime(explanation,TL, sub_results,ttl,min_weight,positive_only=True,save_image=False, num_features=100, hide_rest=False,hide_x_y_ticks=True):
    ''' Fuction to Highlight Positive and Negative Features Provided by LIME-Image
    Args:   
        explanation  :   explanation computed by LIME image Module
        TL           :   Predicted Top Label by LIME-Image Explanation
        savepath     :   Path to Save the Figure
        num_features :   Features to highlight, (Default: 100)
        hide_x_y_ticks:  To Hide or to show X and Y Ticks Parameter (Default: True)    
     '''
    temp, mask = explanation.get_image_and_mask(TL,min_weight=min_weight,positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
    plt.figure(figsize=(3, 3))
    plt.imshow(mark_boundaries(temp/255 / 2 + 0.4, mask))
    if hide_x_y_ticks:
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//ExpByLime'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
#####################################################################################################################
#######################################    EVALUATION FUNCTIONS   ###################################################
#####################################################################################################################
def evaluate_explanation(explanation,X,all_Ys,beta,f_x,RC_Y,r2_score,data_to_csv,model_name):
    ''' 
    This function will evaluate the explanation produced by LIME-Image and will return a dict of keys and values.
    Args:
        explanation  :   explanation computed by LIME image Module 
        X            :   data returned and being used by LIME explain_instance module
        all_Ys       :   labels (for generated data) returned and being used by LIME explain_instance module
        data_to_csv  :   A Initial dictionary having keys and values to be saved in .csv file
        model_name   :   Blackbox Model Name as string being used 
        seg_range    :   Segments Range (25-50,50-100,100-150,150-200)
    '''
    TL = explanation.top_labels[0]    
    Y =all_Ys[:,TL]
    maxval = np.max(np.abs(beta))
    g_x = explanation.local_pred[TL][0]
    r2_score = explanation.score[TL]
    local_pred = explanation.local_pred[TL]
    intercept = explanation.intercept[TL]
    data_to_csv['model_name']     = model_name
    data_to_csv['f_x']            = f_x
    data_to_csv['g_x']            = explanation.local_pred[TL][0]
    data_to_csv['q05_Y']          = np.quantile(Y,0.05)
    data_to_csv['q95_Y']          = np.quantile(Y,0.95)
    data_to_csv['q01_Y']          = np.quantile(Y,0.01)
    data_to_csv['q99_Y']          = np.quantile(Y,0.99)
    data_to_csv['std_Y']          = np.std(Y)
    data_to_csv['std_abs_Y']      = np.std(np.abs(Y))
    data_to_csv['r2']             = r2_score
    data_to_csv['maxval']         = maxval
    data_to_csv['local_pred']     = local_pred
    data_to_csv['intercept']      = intercept
    data_to_csv['std_beta']       = np.std(beta)
    data_to_csv['std_abs_beta']   = np.std(np.abs(beta))
    data_to_csv['mean_beta']      = np.mean(beta) 
    data_to_csv['mean_abs_beta']  = np.mean(np.abs(beta))
    data_to_csv['q05_beta']       = np.quantile(beta,0.05)
    data_to_csv['q95_beta']       = np.quantile(beta,0.95)
    data_to_csv['q01_beta']       = np.quantile(beta,0.01)
    data_to_csv['q99_beta']       = np.quantile(beta,0.99)
    data_to_csv['q25_beta']       = np.quantile(beta,0.25)
    data_to_csv['q75_beta']       = np.quantile(beta,0.75)
    data_to_csv['q10_beta']       = np.quantile(beta,0.10)
    data_to_csv['q90_beta']       = np.quantile(beta,0.90)  
    data_to_csv['max_beta']       = np.max(beta)
    data_to_csv['min_beta']       = np.min(beta)
    data_to_csv['RC_Y']           = RC_Y
    data_to_csv['cv_abs_beta']    = np.std(np.abs(beta)) / np.mean(np.abs(beta))
    data_to_csv['cv_beta']        = np.std(beta) / np.mean(beta)
    
def get_beta_from_expl(explanation=None):
    '''
    Function get_beta_from_expl will compute beta from explanation
    Args:
        expl: Explanation returned by Strtaified Lime Image Explainer
    Result:
        beta: Local Exp for Top Label 
    '''
    if explanation is not None:
        n = len(np.unique(explanation.segments))
        beta = np.zeros(n)
        for i,v in explanation.local_exp[ explanation.top_labels[0] ]:
            beta[i] = v
        return beta
def get_RCY(Y,f_x):
    '''
    Function get_RCY will take two parameters and will compute the InterQuantile Range (IQR(99-1)) divided by f_x
    Args:
        Y          :  Y is the labels returned by Stratified LIME-Image explainer object
        f_x :  f_x is class probability for the image returned by Blackbox model
    Result:
        IQR(99-01)/f_x
    '''
    return (np.quantile(Y,0.99) - np.quantile(Y,0.01)) / f_x 
#####################################################################################################################
#################################################   PAPER FIGURE   ##################################################
#####################################################################################################################
def shapley_p(k, m):
    return 1 / ((k+1) * scipy.special.binom(k, m))

def pdf_bern(k, m, p=0.5):
    return scipy.special.binom(k,m) * (p ** (m)) * ( (1-p) ** (k-m) )

#####################################################################################
def get_CV_beta(beta):
    return np.std(beta) / np.mean(beta)

########################################   COMPUTE EXPLANATION   ####################################

def explanation_module(compute_experiments,files,df_seg,DS_path,sub_results_,result_folder,segs_list,model,model_name,class_names,
                       save_explanations_as_plot,use_stratification,plot_prediction,plot_segments,plot_heatmap,plot_image_mask,
                       hide_color,num_samples,repeat_exp,top_labels,batch_size,lime_image,bb_predict):
    if compute_experiments:
        results_csv = []
        df_data = []
        now = datetime.now()
        print("Example Started\t\t:\t\t\t", now.strftime("%d/%m/%Y %H:%M:%S"))
        for f in files:
            file_name = f'{f+1:08}'
            file = os.path.join(DS_path,'ILSVRC2012_test_'+file_name+'.JPEG')
            
            sub_results = os.path.join(sub_results_,file_name)        
            if not check_folders(sub_results): print(sub_results_,file_name)
    #       Read and resize image according to model Input Layer
            image   = read_process_image(file,model)
            image_arr = np.expand_dims(image,axis = 0)
            predicted = bb_predict(image_arr)
    #         Convert the Predicted into Predicted Class Index (PDI), Class Probability, and Predicted Class Label (PDL)
            (PDI,f_x,PDL) =  get_class_idx_label_score (predicted,class_names)

    #       Plot the blackbox model prediction 
            if plot_prediction:
                plot_save_prediction(image,PDL,f_x,sub_results,file_name,
                                        plot_everything=save_explanations_as_plot,save_image=True)

            df_n = df_seg.loc[(df_seg['filename'] == file_name)]
            for data, row in df_n.T.iteritems():
                filename_seg,md,ks,sr = row.filename,row.max_distance,row.kernal_size,row.target_segs
                segments,segs,segmenter_fn = own_seg(image,md=md,ks=ks)
    #########               Plot the segments Created 
                if plot_segments:
                    plot_seg_image(image,segments,md,ks,sub_results,file_name,save_image=True)
                for hc in hide_color:
                    for us in use_stratification:
                        hcc = 'mean-filled' if hc is None else 'zero-filled'
                        sig = f'{segs}_{hcc}_{us}_{num_samples}'
                        data_to_csv = dict()
                        beta_arr, rcY_arr,r2_arr = [], [], []
#########               Fix Random Seed to make benchmark deterministic and reproducible
                        explainer_lime = lime_image.LimeImageExplainer(random_state=1234)
#########               Create Explanation
                        for repeat in range(repeat_exp):
                            print(repeat+1, end=' ')
                            explanation_ret = explainer_lime.explain_instance(image, bb_predict,
                                                             hide_color=hc,
                                                             top_labels=top_labels,batch_size = batch_size,
                                                             use_stratification = us,num_samples=num_samples,
                                                             segmentation_fn = segmenter_fn,progress_bar=False)
                            X, all_Ys,explanation = explanation_ret #                   split it into 3 variables
                            predicted_cls = explanation.top_labels[0]
                            Y = all_Ys[:, predicted_cls]
                            beta_arr.append(get_beta_from_expl(explanation=explanation))
                            rcY_arr.append(get_RCY(Y, f_x))
                            r2_arr.append(explanation.score[predicted_cls])
                        beta = np.mean(beta_arr, axis=0)
                        RC_Y = np.mean(rcY_arr)
                        r2 = np.mean(r2_arr)
###########################################################################################################################################
#                            Building a Dictionary with Keys and Values to write into Data File  ###########################################################################################################################################
                        data_to_csv = {'filename':str(file_name),'hide_color':str(hcc),'use_stratification':str(us),          'num_samples':str(num_samples),'segments':str(segs),'max_dist':str(md),'kernal_size':str(ks)}
###########################################################################################################################################
#                                     EVALUATING EXPLANATION ###########################################################################################################################################
                        evaluate_explanation(explanation,X,all_Ys,beta,f_x,RC_Y,r2,data_to_csv,model_name)
###########################################################################################################################################
#                                     PLOTTING CLASSIFICATION SCORE
#                       This will generate the Classification Score of Linear Regressor                               ###########################################################################################################################################
                        if plot_classification_score:
                            plot_classification_score_examples(explanation,X,all_Ys,f_x,sub_results,sig,
                                                                  plot_everything=save_explanations_as_plot,save_image=True)
##################################################################################################################################
#                                   PLOT:    HEATMAP
# This will generate heatmap plot based on feature importances computed by us from explanation returned by LIME Image Explainer
#####################################################################################################################################
                        if plot_heatmap:
                            heatmap = heatmap_from_beta(segments, beta)
                            plot_heatmap_lime(heatmap,data_to_csv['maxval'],sub_results,sig,save_result=True,
                                                 show_color_bar=False,color_bar_position='right')
##############################################################################################################################
# PLOT:    GET IMAGE AND MASK BY LIME
# This will generate heatmap plot based on feature importances computed by us from explanation returned by LIME Image Explainer ##############################################################################################################################
                        if plot_image_mask:
                            get_img_mask_lime(explanation,predicted_cls, sub_results,sig,
                                                 min_weight = data_to_csv['maxval']/2,save_image=True, num_features=5, hide_rest=True)
                        results_csv.append(data_to_csv)
                        df_data = pd.DataFrame(results_csv)
                        df_data.to_csv(f'{result_folder}/results_{num_samples}_{files.start+1}_{files.stop}_{segs_list}.csv',
                                       sep = ';', index=False)
                        print(f'{file_name} Segs: {segs} Use_Stratification: {us} CV_abs:{data_to_csv["cv_abs_beta"]:0.5} CV: {data_to_csv["cv_beta"]:0.5}')
        now = datetime.now()
        print("Example Completed\t\t:\t\t\t", now.strftime("%d/%m/%Y %H:%M:%S"))
        return df_data
    else:
        if os.path.exists(f'{result_folder}/results_{num_samples}_{files.start+1}_{files.stop}_{segs_list}.csv'):
            df_data = pd.read_csv(f'{result_folder}/results_{num_samples}_{files.start+1}_{files.stop}_{segs_list}.csv', sep = ';')
            return df_data
        else:
            print("Data File Doesn't Exit")