#!/usr/bin/python
import sys
sys.path.insert(0, '/Users/ragav/GitHub/Convolutional-Neural-Networks/')


from samosa.cnn import cnn_mlp
from samosa.core import ReLU  
from samosa.util import load_network
from samosa.dataset import setup_dataset
import os

import pdb

def run_cnn( 
                    arch_params,
                    optimization_params ,
                    dataset, 
                    filename_params,
                    visual_params,
                    n_epochs = 50,
                    ft_epochs = 200, 
                    validate_after_epochs = 1,
                    verbose = False, 
           ):            
               
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose  )   
               
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )          
    net.build_network(verbose = verbose)                         
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )          
    net.test( verbose = verbose )                      
    net.save_network ()   
     
# this is for the generality experiments. This retrains by freezing and unfreezing layers diferenty.     
def generality_experiment(                
                    arch_params,
                    optimization_params ,
                    dataset, 
                    original_filename_params,   
                    filename_params_retrain, 
                    retrain_params,     
                    visual_params,    
                    validate_after_epochs = 1,  
                    n_epochs = 50,
                    ft_epochs = 200, 
                    verbose = False                                                                                                                              

                      ):         
                                                                                                     
    params_loaded, arch_params_loaded = load_network (original_filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False)   
 
    # retrain is used to do the dataset some wierd experiments.     
    retrain_net = cnn_mlp(  filename_params = filename_params_retrain,
                            arch_params = arch_params,
                            optimization_params = optimization_params,
                            retrain_params = retrain_params,
                            init_params = params_loaded,
                            verbose =verbose   )  
                                                
    retrain_net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )      
    retrain_net.build_network ( verbose = verbose )                           
    retrain_net.train ( n_epochs = n_epochs, 
                        ft_epochs = ft_epochs,
                         validate_after_epochs = validate_after_epochs,
                         verbose = verbose)                                                 
    retrain_net.test ( verbose = verbose )       
    retrain_net.save_network ()                       
                           
    
## Boiler Plate ## 
if __name__ == '__main__':
    """         
    if os.path.isfile('dump.txt'):
        f = open('dump.txt', 'a')
    else:
        f = open('dump.txt', 'w')
        f.close()
        f.open ('dump.txt','a')
        
    f.write("... main net")
    """
    dataset = "_datasets/_dataset_41703"    
    retrain_dataset = "_datasets/_dataset_39440"

    # run the base CNN as usual.              
    filename_params = { 
                        "results_file_name"     : "../results/results.txt",      
                        "error_file_name"       : "../results/error.txt",
                        "cost_file_name"        : "../results/cost.txt",
                        "confusion_file_name"   : "../results/confusion.txt",
                        "network_save_name"     : "../saved_results/colon-colon/results/network.pkl.gz "
                    }
                    
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 20,
                        "n_visual_images"       : 64,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.85,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,
			                "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,    
                            }        

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : True,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5, 0.5 ],
                    "num_nodes"                         : [ 1024 ],                                     
                    "outs"                              : 2,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU , ReLU,  ReLU, ReLU, ReLU ],             
                    "cnn_batch_norm"                    : [ True , True,  True, True, True],
                    "mlp_batch_norm"                    : True,
                    "nkerns"                            : [     36  ,     36    , 64  ,   96   , 64],              
                    "filter_size"                       : [ ( 5, 5 ) , (5, 5 ), (5, 5), (5, 5) , (5, 5)],
                    "pooling_size"                      : [ ( 2, 2 ) , (2, 2 ), (2, 2), (1, 1), (1, 1) ],
                    "conv_stride_size"                  : [ ( 1, 1 ) , (1, 1 ), (1, 1), (1, 1) , (1, 1)],
                    "cnn_maxout"                        : [  1,         1 ,       1,     1 ,       1],                    
                    "mlp_maxout"                        : [  1    ],
                    "cnn_dropout_rates"                 : [ 0.5,        0.5     , 0.5 ,  0.5,   0.5   ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "use_bias"                          : True,
                    "max_out"                           : 0 

                 }                          
    
    # other loose parameters. 
    n_epochs = 2
    validate_after_epochs = 1
    ft_epochs = 2
    verbose = False 

    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = dataset, 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,                                                
                )
                
                
                                                               
                
    # All frozen
      
    arch_params ["outs"]  = 102
    
    print "... running all frozen"
    f.write("... running all frozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, True, True, True, True, True, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_0.txt",      
                        "error_file_name"       : "../results/error_retrain_0.txt",
                        "cost_file_name"        : "../results/cost_retrain_0.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_0.txt",
                        "network_save_name"     : "../results/network_retrain_0.pkl.gz "
                    }                    
                                      
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                       
                      
                     
                      
    # one Unfrozen 
    f.write("... running one unfrozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, True, True, True, True, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_1.txt",      
                        "error_file_name"       : "../results/error_retrain_1.txt",
                        "cost_file_name"        : "../results/cost_retrain_1.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_1.txt",
                        "network_save_name"     : "../results/network_retrain_1.pkl.gz "
                    }                    
                                       
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                       
                      
                      
                      
                      
                      
                     
    # two Unfrozen 
    print "... running two unfrozen"
    f.write("... running two unfrozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, True, True, True, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_2.txt",      
                        "error_file_name"       : "../results/error_retrain_2.txt",
                        "cost_file_name"        : "../results/cost_retrain_2.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_2.txt",
                        "network_save_name"     : "../results/network_retrain_2.pkl.gz "
                    }                    
                                     
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                            
                      
                      
                      
                      
                      
                      
                      
    # three Unfrozen 
    print "... running three unfrozen"
    f.write("... running three unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, True, True, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_3.txt",      
                        "error_file_name"       : "../results/error_retrain_3.txt",
                        "cost_file_name"        : "../results/cost_retrain_3.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_3.txt",
                        "network_save_name"     : "../results/network_retrain_3.pkl.gz "
                    }                    
                    
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                  
                      

                      
                      
                      
                      
                      
                      
    # four Unfrozen 
    print "... running four unfrozen"
    f.write("... running four unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, True, False, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_4.txt",      
                        "error_file_name"       : "../results/error_retrain_4.txt",
                        "cost_file_name"        : "../results/cost_retrain_4.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_4.txt",
                        "network_save_name"     : "../results/network_retrain_4.pkl.gz "
                    }                    
                    

    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                  
                      
                      
                      
                      
                      
                      
                      
                      
                      
    # five Unfrozen 
    print "... running five unfrozen"
    f.write("... running five unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ True, False, False, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_5.txt",      
                        "error_file_name"       : "../results/error_retrain_5.txt",
                        "cost_file_name"        : "../results/cost_retrain_5.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_5.txt",
                        "network_save_name"     : "../results/network_retrain_5.pkl.gz "
                    }                    
                                     
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                  
                      





    # five Unfrozen 
    print "... running all unfrozen"
    f.write("... running all unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, False ],
                        "freeze"            : [ False, False, False, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_6.txt",      
                        "error_file_name"       : "../results/error_retrain_6.txt",
                        "cost_file_name"        : "../results/cost_retrain_6.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_6.txt",
                        "network_save_name"     : "../results/network_retrain_6.pkl.gz "
                    }                    
                                 
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                    
                      
                      
    pdb.set_trace()                             