
import torch 
from evaluation import Evaluator
from dataset import Dataset, mean_transform, std_transform
from feature_extractor import FeatureExtractor
import numpy as np

from sklearn.model_selection import train_test_split

SEED = 42 ### set for reproducibility


class ThresholdOptDataset():
    """
    Constructing 4 threshold optimization sets and 1 threshold evaluation set for the experiments. 
    
    The available component datasets are ID dataset (defined by id_dataset), one Target Threat dataset (defined by ood_dataset), and 
    the remaining eight threat sets. For each set to construct (5 sets), different information of the data included in each set are extracted 
    (features from a layer, logits, softmax, prediction class, ground truth class, ...). The extractor based on the network architectures (either DenseNet or ResNet), 
    on the layer examined (defined by layer_id). The four strategies of threshold optimizations sets are denoted case 0, case 1 , case 2, case 3 
    (respectively corresponding to strategy ID, ID+T, ID+T+O, ID+O).
            
    Args:
        id_dataset (str): Name of the in-distribution dataset.
        ood_dataset (str): Name of the Target Threat dataset.
        network (str): Network that will be used to process the dataset.
        layer_id (int): Layer ID examined.
        cov_shift (str or None): transform applied to Target Threat before going through the network.
        atk (str or None): attack applied to Target Threat before going through the network.
        batch_size (int): Batch size used to process the dataset.
        split_ratio (float): split ratio of original test split from ID set, and of Target Threat, will be 0.5.
        save_flag_type (boolean): if one want to know threat type (novelty, cov shift, atk) of each data in each set in final.
        
    """
    
    novelty_dataset= {'cifar10': ['cifar100', 'lsun', 'svhn'],
                      'cifar100': ['cifar10', 'lsun', 'svhn'],
                      'svhn': ['cifar10', 'lsun', 'tiny_imagenet']} #same settings as in Out-Of-Distribution Detection Is Not All You Need

    accepted_transforms = ["brightness", "blur", "pixelization"] #covariate shift
    accepted_attacks = ["fgsm", "deepfool", "pgd"] #adversarial attack

    
    def __init__(self, id_dataset, ood_dataset, network, layer_id, split_ratio, cov_shift, atk, batch_size, beta=None, save_flag_type=False, device_name=None):
        
        # Public attributes 
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.network = network
        self.layer_id = layer_id
        self.split_ratio = split_ratio
        self.cov_shift = cov_shift
        self.atk = atk            
        self.beta = beta
        self.batch_size = batch_size
        self.SEED = SEED
        
        if device_name is None:
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_name = device_name
        self._device = torch.device(self.device_name)
        
        self.feature_extractor = FeatureExtractor(self.network, self.id_dataset, [self.layer_id], self.device_name)

        # Ensure dataset is valid
        self._check_accepted_dataset_expThreshOpt() 
        self._check_accepted_network() 
        self._check_accepted_transforms_expThreshOpt() 
        self._check_accepted_attack()
        
        # Import data od id and main ood dataset
        self.dataset_train = Dataset(self.id_dataset, "train", self.network, batch_size=self.batch_size)

        self.dataset_test = Dataset(self.id_dataset, "test", self.network, None, None, batch_size=self.batch_size)
        self.dataset_ood = Dataset(self.ood_dataset, "test", self.network, 
                                   additional_transform=self.cov_shift, adversarial_attack=self.atk, batch_size=self.batch_size)
        
        # List of other OOD datasets 
        self._get_name_otherOODdataset()
        
        # Construct optimization and test set for all cases
        self._fitting_classifier_trainingset() ###
        
        self.save_flag_type = save_flag_type
        self._construct_evaluationset()
        self._construct_optimizationset_case0()
        self._construct_optimizationset_case1()
        self._construct_optimizationset_case2()
        self._construct_optimizationset_case3()
    
    
    def _get_name_otherOODdataset(self):
        other_novelty = [nov for nov in self.novelty_dataset[self.id_dataset] if nov != self.ood_dataset]
        other_covshift = [shift for shift in self.accepted_transforms if shift != self.cov_shift]
        other_attack = [atk for atk in self.accepted_attacks if atk != self.atk]
        
        self.other_ood_dataset = other_novelty + other_covshift + other_attack
    
    def _fitting_classifier_trainingset(self):
        # Fitting monitor with classifier training set
        self.features_train, self.logits_train, self.softmax_train, \
            self.pred_train, self.lab_train = self.feature_extractor.get_features(self.dataset_train)
        
        
    def _construct_evaluationset(self):
        
        ### Prepare evaluation samples from id dataset
        # Extract feature from layer for data_test of model and data_oms
        features_test, logits_test, softmax_test, \
            pred_test, lab_test = self.feature_extractor.get_features(self.dataset_test)
        # Calculate ground truth of data in case of monitor; y_true = 0 if accepted, 1 if rejected
        eval_oms_id = Evaluator("oms", is_novelty=False)
        eval_oms_id.fit_ground_truth(lab_test, np.array([]), pred_test,  np.array([]))
        
        ### Prepare evaluation samples from ood dataset
        # Extract feature from layer for data_test of model and data_oms
        features_ood, logits_ood, softmax_ood, \
            pred_ood, lab_ood = self.feature_extractor.get_features(self.dataset_ood)
        # Calculate ground truth of data in case of monitor; y_true = 0 if accepted, 1 if rejected
        eval_oms_ood = Evaluator("oms", is_novelty=self.id_dataset!=self.ood_dataset)
        eval_oms_ood.fit_ground_truth(np.array([]), lab_ood, np.array([]), pred_ood)
        
        if self.id_dataset != self.ood_dataset: #ood novelty in evaluation
            # Split id data in optimization and evaluation set, with shuffle=True and split_ratio previously defined
            self.features_test_p1, self.features_test_p2, \
            self.logits_test_p1, self.logits_test_p2, \
            self.softmax_test_p1, self.softmax_test_p2, \
            self.pred_test_p1, self.pred_test_p2, \
            self.lab_test_p1, self.lab_test_p2, \
            self.y_true_id_p1, self.y_true_id_p2= train_test_split(features_test[0], ### for one layer
                                                    logits_test,
                                                    softmax_test,
                                                    pred_test,
                                                    lab_test,
                                                    eval_oms_id.y_true,
                                                    test_size=1-self.split_ratio,
                                                    random_state=self.SEED,
                                                    shuffle=True)


            # Split id data in optimization and evaluation set, with shuffle=True and split_ratio previously defined
            self.features_ood_p1, self.features_ood_p2, \
            self.logits_ood_p1, self.logits_ood_p2, \
            self.softmax_ood_p1, self.softmax_ood_p2, \
            self.pred_ood_p1, self.pred_ood_p2, \
            self.lab_ood_p1, self.lab_ood_p2, \
            self.y_true_ood_p1, self.y_true_ood_p2= train_test_split(features_ood[0], ### for one layer
                                                    logits_ood,
                                                    softmax_ood,
                                                    pred_ood,
                                                    lab_ood,
                                                    eval_oms_ood.y_true,
                                                    test_size=1-self.split_ratio,
                                                    random_state=self.SEED,
                                                    shuffle=True)
        else: 
            #split id data and ood data the same way into optimization and evaluation set 
            self.features_test_p1, self.features_test_p2, \
            self.logits_test_p1, self.logits_test_p2, \
            self.softmax_test_p1, self.softmax_test_p2, \
            self.pred_test_p1, self.pred_test_p2, \
            self.lab_test_p1, self.lab_test_p2, \
            self.y_true_id_p1, self.y_true_id_p2, \
            \
            self.features_ood_p1, self.features_ood_p2, \
            self.logits_ood_p1, self.logits_ood_p2, \
            self.softmax_ood_p1, self.softmax_ood_p2, \
            self.pred_ood_p1, self.pred_ood_p2, \
            self.lab_ood_p1, self.lab_ood_p2, \
            self.y_true_ood_p1, self.y_true_ood_p2= train_test_split(features_test[0], ### for one layer
                                                    logits_test,
                                                    softmax_test,
                                                    pred_test,
                                                    lab_test,
                                                    eval_oms_id.y_true,
                                                    features_ood[0], ### for one layer
                                                    logits_ood,
                                                    softmax_ood,
                                                    pred_ood,
                                                    lab_ood,
                                                    eval_oms_ood.y_true,
                                                    test_size=1-self.split_ratio,
                                                    random_state=self.SEED,
                                                    shuffle=True)
            
        
        # Reformat features into list of arrays
        self.features_test_p1 = [self.features_test_p1]
        self.features_test_p2 = [self.features_test_p2]

        self.features_ood_p1 = [self.features_ood_p1]
        self.features_ood_p2 = [self.features_ood_p2]    
        
        
        ### Concatenating id samples and ood samples to form evaluation set 
        self.features_allcase_evaluation = [np.concatenate((self.features_test_p2[0], self.features_ood_p2[0]), axis=0)]
        self.logits_allcase_evaluation = np.concatenate((self.logits_test_p2, self.logits_ood_p2), axis=0)
        self.softmax_allcase_evaluation = np.concatenate((self.softmax_test_p2, self.softmax_ood_p2), axis=0)
        self.pred_allcase_evaluation = np.concatenate((self.pred_test_p2, self.pred_ood_p2), axis=0)
        self.lab_allcase_evaluation = np.concatenate((self.lab_test_p2, self.lab_ood_p2), axis=0)
        self.y_true_allcase_evaluation = np.concatenate((self.y_true_id_p2, self.y_true_ood_p2), axis=0)
        
    
    def _construct_optimizationset_case0(self):
        self.features_case0_optimization = self.features_test_p1.copy()
        self.logits_case0_optimization = self.logits_test_p1.copy()
        self.softmax_case0_optimization = self.softmax_test_p1.copy()
        self.pred_case0_optimization = self.pred_test_p1.copy()
        self.lab_case0_optimization = self.lab_test_p1.copy()
        self.y_true_case0_optimization = self.y_true_id_p1.copy()    
        
        
    def _construct_optimizationset_case1(self):
        # Concatenate 2 datasets, data_test of model and data_oms into one
        self.features_case1_optimization = [np.concatenate((self.features_test_p1[0], self.features_ood_p1[0]), axis=0)]
        self.logits_case1_optimization = np.concatenate((self.logits_test_p1, self.logits_ood_p1), axis=0)
        self.softmax_case1_optimization = np.concatenate((self.softmax_test_p1, self.softmax_ood_p1), axis=0)
        self.pred_case1_optimization = np.concatenate((self.pred_test_p1, self.pred_ood_p1), axis=0)
        self.lab_case1_optimization = np.concatenate((self.lab_test_p1, self.lab_ood_p1), axis=0)
        self.y_true_case1_optimization = np.concatenate((self.y_true_id_p1, self.y_true_ood_p1), axis=0)    
        
    def _construct_optimizationset_case2(self):
        
        ### Optimization set of case 2 is concatenation of optimization set of case 1 with additional OOD datasets
        # Initialize first items of optimization set
        self.features_case2_optimization = self.features_case1_optimization.copy()
        self.logits_case2_optimization = self.logits_case1_optimization.copy()
        self.softmax_case2_optimization = self.softmax_case1_optimization.copy()
        self.pred_case2_optimization = self.pred_case1_optimization.copy()
        self.lab_case2_optimization = self.lab_case1_optimization.copy()
        self.y_true_case2_optimization = self.y_true_case1_optimization.copy()
        
        if self.save_flag_type:
            flag_id_case1_optimization = ["ID" for _ in range(len(self.y_true_id_p1))]
            
            if self.id_dataset != self.ood_dataset:
                flag_ood_case1_optimization = [f"Novelty_Target" for _ in range(len(self.y_true_ood_p1))]
            else: 
                flag_ood_case1_optimization = [f"{(self.atk or '') + (self.cov_shift or '') }_Target" for _ in range(len(self.y_true_ood_p2))]
            self.flag_type_case1_optimization = flag_id_case1_optimization + flag_ood_case1_optimization
            self.flag_type_case2_optimization = self.flag_type_case1_optimization.copy()

        # Concatenate next items (different ood dataset) into optimization set
        for other_ood_name in self.other_ood_dataset:
            # Novelty set up
            if other_ood_name in self.novelty_dataset[self.id_dataset]:
                ood_dataset_tmp = other_ood_name
                cov_shift_tmp = None
                atk_tmp = None
                is_novelty = True
                
            # Covariate shift set up
            elif other_ood_name in self.accepted_transforms: 
                ood_dataset_tmp = self.id_dataset
                cov_shift_tmp = other_ood_name
                is_novelty = False
            # Adversarial attack set up
            else:
                cov_shift_tmp = None
                atk_tmp = other_ood_name

            dataset_ood_tmp = Dataset(ood_dataset_tmp, "test", self.network, cov_shift_tmp, atk_tmp, batch_size=self.batch_size)
            features_ood_tmp, logits_ood_tmp, softmax_ood_tmp, \
                pred_ood_tmp, lab_ood_tmp = self.feature_extractor.get_features(dataset_ood_tmp)

            # Calculate ground truth of data in case of monitor; y_true = 0 if accepted, 1 if rejected
            eval_oms = Evaluator("oms", is_novelty=is_novelty)
            eval_oms.fit_ground_truth(np.array([]), lab_ood_tmp,  np.array([]), pred_ood_tmp)

            # Concatenate iteratively datasets, into optimization set
            self.features_case2_optimization = [np.concatenate((self.features_case2_optimization[0], features_ood_tmp[0]), axis=0)]
            self.logits_case2_optimization = np.concatenate((self.logits_case2_optimization, logits_ood_tmp), axis=0)
            self.softmax_case2_optimization = np.concatenate((self.softmax_case2_optimization, softmax_ood_tmp), axis=0)
            self.pred_case2_optimization = np.concatenate((self.pred_case2_optimization, pred_ood_tmp), axis=0)
            self.lab_case2_optimization = np.concatenate((self.lab_case2_optimization, lab_ood_tmp), axis=0)
            self.y_true_case2_optimization = np.concatenate((self.y_true_case2_optimization, eval_oms.y_true), axis=0)
            
            if self.save_flag_type: 
                if other_ood_name in self.novelty_dataset[self.id_dataset]:
                    flag_ood_tmp =  [f"Novelty_{other_ood_name}(Other)" for _ in range(len(eval_oms.y_true))]
                else: 
                    flag_ood_tmp =  [f"{other_ood_name}_Other" for _ in range(len(eval_oms.y_true))]
                self.flag_type_case2_optimization += flag_ood_tmp
                
    def _construct_optimizationset_case3(self):
        
        ### optimization set of case 3 is concatenation of optimization set of case 0 with additional OOD datasets
        # Initialize first items of optimization set
        self.features_case3_optimization = self.features_case0_optimization.copy()
        self.logits_case3_optimization = self.logits_case0_optimization.copy()
        self.softmax_case3_optimization = self.softmax_case0_optimization.copy()
        self.pred_case3_optimization = self.pred_case0_optimization.copy()
        self.lab_case3_optimization = self.lab_case0_optimization.copy()
        self.y_true_case3_optimization = self.y_true_case0_optimization.copy()
        
        if self.save_flag_type:
            self.flag_type_case0_optimization = ["ID" for _ in range(len(self.y_true_id_p1))]
            self.flag_type_case3_optimization = self.flag_type_case0_optimization.copy()
            
        # Concatenate next items (different ood dataset) into optimization set
        for other_ood_name in self.other_ood_dataset:
            # Novelty set up
            if other_ood_name in self.novelty_dataset[self.id_dataset]:
                ood_dataset_tmp = other_ood_name
                cov_shift_tmp = None
                atk_tmp = None
                is_novelty = True
            # Covariate shift set up
            elif other_ood_name in self.accepted_transforms: 
                ood_dataset_tmp = self.id_dataset
                cov_shift_tmp = other_ood_name
                is_novelty = False
            # Adversarial attack set up
            else:
                cov_shift_tmp = None
                atk_tmp = other_ood_name

            dataset_ood_tmp = Dataset(ood_dataset_tmp, "test", self.network, cov_shift_tmp, atk_tmp, batch_size=self.batch_size)
            features_ood_tmp, logits_ood_tmp, softmax_ood_tmp, \
                pred_ood_tmp, lab_ood_tmp = self.feature_extractor.get_features(dataset_ood_tmp)

            # Calculate ground truth of data in case of monitor; y_true = 0 if accepted, 1 if rejected
            eval_oms = Evaluator("oms", is_novelty=is_novelty)
            eval_oms.fit_ground_truth(np.array([]), lab_ood_tmp,  np.array([]), pred_ood_tmp)

            # Concatenate iteratively datasets, into optimization set
            self.features_case3_optimization = [np.concatenate((self.features_case3_optimization[0], features_ood_tmp[0]), axis=0)]
            self.logits_case3_optimization = np.concatenate((self.logits_case3_optimization, logits_ood_tmp), axis=0)
            self.softmax_case3_optimization = np.concatenate((self.softmax_case3_optimization, softmax_ood_tmp), axis=0)
            self.pred_case3_optimization = np.concatenate((self.pred_case3_optimization, pred_ood_tmp), axis=0)
            self.lab_case3_optimization = np.concatenate((self.lab_case3_optimization, lab_ood_tmp), axis=0)
            self.y_true_case3_optimization = np.concatenate((self.y_true_case3_optimization, eval_oms.y_true), axis=0) 
           
            if self.save_flag_type: 
                if other_ood_name in self.novelty_dataset[self.id_dataset]:
                    flag_ood_tmp =  [f"Novelty_{other_ood_name}(Other)" for _ in range(len(eval_oms.y_true))]
                else: 
                    flag_ood_tmp =  [f"{other_ood_name}_Other" for _ in range(len(eval_oms.y_true))]
                self.flag_type_case3_optimization += flag_ood_tmp
            
    def _check_accepted_dataset_expThreshOpt(self):
        """Ensures that the queried dataset is valid."""
        if self.id_dataset not in self.novelty_dataset.keys():
            raise ValueError("Accepted dataset pairs are: %s" % str(list(novelty_dataset.keys()))[1:-1] + "_test")
        if self.atk != None or self.cov_shift != None:
            assert self.ood_dataset == self.id_dataset
        elif self.atk != None and self.cov_shift != None:
            raise Exception("Class not support usage of both cov shift and adv attack on single dataset!")

    def _check_accepted_network(self):
        """Ensures that the queried neural network is valid."""
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_transforms_expThreshOpt(self):
        """Ensures that the queried transform is valid."""
        if self.cov_shift not in self.accepted_transforms + [None]:
            raise ValueError("Accepted data transforms are: %s" % str(accepted_transforms)[1:-1])

    def _check_accepted_attack(self):
        """Ensures that the queried adversarial attack is valid."""
        if self.atk not in self.accepted_attacks + [None]:
            raise ValueError("Accepted attacks are: %s" % str(accepted_attacks)[1:-1])
            