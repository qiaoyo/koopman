from ntpath import exists
import scipy.io as sio
import numpy as np
import os
def load_mat_file(file_path):
    """
    读取.mat文件并转换为Python字典格式
    
    参数:
        file_path: .mat文件的路径
        
    返回:
        转换后的Python字典
    """
    try:
        # 读取.mat文件
        mat_data = sio.loadmat(file_path)
        
        # 移除特殊键（MATLAB自动生成的元数据）
        for key in ['__header__', '__version__', '__globals__']:
            if key in mat_data:
                del mat_data[key]
        
        return mat_data
        
    except Exception as e:
        print(f"读取.mat文件时出错: {str(e)}")
        return None

def convert_mat_to_numpy(mat_data,save_path):
    """
    将.mat数据转换为numpy数组格式
    
    参数:
        mat_data: 从.mat文件读取的数据字典
        
    返回:
        转换后的numpy数组字典
    """
    numpy_dict = {}
    
    for key, value in mat_data.items():
        # 对mean_values特殊处理
        if key == 'mean_values'or key=='max_values':
            # 获取结构化数组的数据
            new_dict = {}
            struct_array = value[0][0]
            # 从dtype中获取字段名称
            field_names = struct_array.dtype.names
            # 根据字段名称创建对应的字典
            for field_name in field_names:
                new_dict[field_name] = np.array(struct_array[field_name][0])
            numpy_dict[key] = new_dict
            np.save(os.path.join(save_path,key+'.npy'),new_dict)
        # 对其他数组的处理保持不变
        if key=='flights':
            struct_array = value[0]
            print(struct_array.shape)
            for i,struct_data in enumerate(struct_array):
                os.makedirs(os.path.join(save_path,key,str(i)),exist_ok=True)
                struct_names= struct_data.dtype.names
                for struct_name in struct_names:
                    np.save(os.path.join(save_path,key,str(i),struct_name+'.npy'),np.array(struct_data[0][0][struct_name]))
                    print(struct_data[0][0][struct_name].shape,struct_name)
                # struct_names= struct_data.dtype.names
                # print(struct_names)
                # break
    
    return numpy_dict

def mat2npy():
    file_path = r"C:\Users\Administrator\Desktop\koopman-data\data\AscTec_Pelican_Flight_Dataset.mat"
    save_path=r"C:\Users\Administrator\Desktop\koopman-data\data"
    # 读取.mat文件
    mat_data = load_mat_file(file_path)
    
    if mat_data is not None:
        # 打印原始数据的键
        print("Mat文件中的变量:")
        for key in mat_data.keys():
            print(f"- {key}: {type(mat_data[key])}")
        
        # 转换为numpy格式
        numpy_data = convert_mat_to_numpy(mat_data,save_path)
        
        # 打印转换后的数据信息
        print("\n转换后的数据:")
        for key, value in numpy_data.items():
            print(f"- {key}: value: {value}")

def check_npy():
    fliight_len=[]

    file_path=r"C:\Users\Administrator\Desktop\koopman-data\data\max_values.npy"
    data=np.load(file_path,allow_pickle=True).item()
    print(data.keys())
    for key,value in data.items():
        print(key,value.shape)

    file_path= r"C:\Users\Administrator\Desktop\koopman-data\data\mean_values.npy"
    data=np.load(file_path,allow_pickle=True).item()
    print(data.keys())
    for key,value in data.items():
        print(key,value.shape)

    # 遍历所有flight文件夹（0-53）
    for folder_idx in range(54):
        # 构建文件夹路径
        folder_path = os.path.join(r"C:\Users\Administrator\Desktop\koopman-data\data\flights", str(folder_idx))
        if os.path.exists(folder_path):
            # 遍历该文件夹下的所有.npy文件
            '''
            ["Pos", "Vel", "Euler", "Euler_Rates", "pqr", "Motors", "Motors_CMD","len"]
            Pos (3,)
            Vel (3,)
            Euler (3,)
            Euler_Rates (3,)
            pqr (3,)
            Motors (4,)
            Motors_CMD (4,)
            len 1
            '''
            FILE_NAME=["Pos", "Vel", "Euler", "Euler_Rates", "pqr", "Motors", "Motors_CMD"]
            length=np.load(os.path.join(folder_path,'len.npy'))
            fliight_len.append(length)
            for file_name in FILE_NAME:
                file_path = os.path.join(folder_path, file_name + '.npy')
                data = np.load(file_path)
                print(f"Folder: {folder_idx}, File: {file_name}, DataShape: {data.shape}",length)
                try:
                    assert data.shape[0]>=length
                except:
                    print(folder_idx,file_name,data.shape[0],length)
    return fliight_len 

                    
# 使用示例
def read_flight_49():
    """
    读取flights/49文件夹下的所有npy文件并显示其形状
    """
    folder_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights\49"
    
    # 定义要读取的文件名列表
    file_names = ["Pos", "Vel", "Euler", "Euler_Rates", "pqr", "Motors", "Motors_CMD", "len"]
    
    print("Flight 49数据形状：")
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name + '.npy')
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(f"{file_name}: {data.shape}")
            if file_name!='len':    
                print(data[0:5])
        else:
            print(f"{file_name}文件不存在")

if __name__ == "__main__":
    read_flight_49()
    # flight_len=check_npy()
    # for i,data in enumerate(flight_len):
        # print(i,data)