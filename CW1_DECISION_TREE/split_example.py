one_fold_data = []
nine_folds_data = []
            
def split10Fold(data, one_fold_data, nine_folds_data, time):
    array = np.array(data)
    num_of_data = array.shape[0]
    one_fold = num_of_data // 10
    nine_fold = num_of_data - one_fold
    
   ## print(one_fold)
   ## print(nine_fold)
   
    start = (time - 1) * one_fold
   
    end = start + one_fold
    
    print(start)
    print(end)
    
    
    for i in range(0, num_of_data):
        if(i < end and i >= start):
            one_fold_data.append(data[i])
        else:
            nine_folds_data.append(data[i])
            
            
            
    one_fold_data = np.asarray(one_fold_data)
    nine_folds_data = np.asanyarray(nine_folds_data)
    
    
split10Fold(data_x, one_fold_data, nine_folds_data, 3)