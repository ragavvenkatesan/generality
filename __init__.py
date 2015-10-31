#!/usr/bin/python
from samosa.build import network
from samosa.core import ReLU, Sigmoid, Softmax, Tanh, Identity
from samosa.util import load_network
from samosa.dataset import setup_dataset

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
    net = network(  filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )   
               
    net.init_data ( dataset, outs = arch_params["outs"], verbose = verbose )      
    
    net.build_network(   arch_params = arch_params,
                         optimization_params = optimization_params,
                         reload_params = False,
                         verbose = verbose)
                         
    net.create_dirs ( visual_params = visual_params )   
                       
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )          
    net.test( verbose = verbose )
   
                              
    net.save_network ()   
    
def generality_experiments(                
                    arch_params,
                    optimization_params ,
                    dataset, 
                    filename_params,
                    visual_params,
                    filename_params_retrain,  
                    data_params_retrain,
                    retrain_params,                  
                    n_epochs = 50,
                    ft_epochs = 200, 
                    validate_after_epochs = 1,                    
                    verbose = False
                      ):         
                                                    
    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = True)   
 
    # retrain is used to do the dataset some wierd experiments.     
    retrain_net = network( 
                     data_params = data_params_retrain,
                     filename_params = filename_params_retrain,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )  
    
    retrain_net.init_data ( dataset )   
    retrain_net.build_network (
                           arch_params = arch_params_loaded,
                           optimization_params = optimization_params,
                           init_params = params_loaded,
                           retrain_params = reload_params,
                           verbose = verbose )    
                           
    retrain_net.train ( n_epochs = n_epochs, 
                        ft_epochs = ft_epochs,
                         validate_after_epochs = validate_after_epochs,
                         verbose = verbose)                                             
    
    retrain_net.test ( verbose = verbose )                         
                    
    pdb.set_trace()               
    
## Boiler Plate ## 
if __name__ == '__main__':
             
    # run the base CNN as usual.              
    filename_params = { 
                        "results_file_name"     : "../results/results.txt",      
                        "error_file_name"       : "../results/errortxt",
                        "cost_file_name"        : "../results/cost.txt",
                        "confusion_file_name"   : "../results/confusion.txt",
                        "network_save_name"     : "../results/network.pkl.gz "
                    }
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 81,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.65,
                            "mom_interval"                      : 50,
                            "mom_type"                          : 0,                         
                            "initial_learning_rate"             : 0.01,
			                "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : False,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 0,    
                            }        

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : False,
                    "mlp_dropout_rates"                 : [ 0.5,  0.5 ],
                    "num_nodes"                         : [ 400 ],                                     
                    "outs"                              : 8,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU , ReLU ],             
                    "cnn_batch_norm"                    : [ True , True ],
                    "mlp_batch_norm"                    : True,
                    "nkerns"                            : [     20,     20 ],              
                    "filter_size"                       : [ ( 3, 5, 5 ) , (5, 5 , 5)],
                    "pooling_size"                      : [ ( 2, 2, 2 ) , (1, 2, 2 )],
                    "conv_stride_size"                  : [ ( 1, 1, 1 ) , (1, 1 , 1 )],
                    "cnn_maxout"                        : [  2,         1 ],                    
                    "mlp_maxout"                        : [  1    ],
                    "cnn_dropout_rates"                 : [ 0.5,     0.5    ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "max_out"                           : 0 

                 }           
                 
    # needed only if you are reloading a network.                  
    reload_params = {
                        "copy_from_old"     : [ True, True ],
                        "freeze"            : [ False, True ]
                    }

    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = "_datasets/_dataset_36462", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = 1,
                    n_epochs                = 50,
                    ft_epochs               = 100, 
                    verbose                 = False ,                                                
                )