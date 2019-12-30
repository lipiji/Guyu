#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: myanzhang
# @Date:   2019-08-20

'''
配置文件版本号
'''
version = '1.1.3'

'''
用户的个人Token, 请在http://jizhi.oa.com个人头像位置点击用户名获取
'''
Token = 'mHkPsVIRxMxoFV_FT208mw'

'''
创建任务，参数配置
'''
task_params = {
  # 任务类型, tensorflow/general_gpu_type
  'task_type': 'gpt2',

  'common': {
    # 业务唯一标识         
    'business_flag': 'nlpc', 

    # 备注字段，对任务的补充说明，如果不填，则传''空串
    'description_info': '', 

    # 任务名称，方便用户辨识的名称，最长不超过128字符
    'readable_name': 'gpt2',

     # 任务标识，用于全局唯一标识一个任务。 提示：只可以含有大小写字母、数字、下划线和连字符。后缀自动加时间标志
    'task_flag': 'gpt2',
    
    # 特别说明：dataset_id 和 dataset_params两个字段：
    # 当用户需要创建并使用新的数据集，则只需填写 dataset_params 字段，dataset_id 字段值为 ''
    # 当用户使用已有的数据集，则填写 dataset_id 字段，使该字段不为''
    # 数据集id
    'dataset_id': '', 
    'dataset_params': {                          
      # 数据集来源, local_upload/plat_ceph/outer_ceph
      'dataset_source': 'local_upload', 

      # 数据集名称，自定义名称； 注意：若要修改数据集，该字段可以修改
      'dataset_name': 'pjdata', 

      'path_info': {
        # 若为local_upload: 数据集的本地路径； 若为plat_ceph/outer_ceph: 数据集在ceph的路径          
        'path': './data', 

        # 数据集解析文件的本地路径, "通用任务" 不需要填写该字段值; 注意：若要修改数据集，该字段可以修改
        'dataset_code_path': '', 

        # 业务私有ceph集群的addr:<ip:port>，仅当 数据集来源 方式为outer_ceph为必填字段，其余方式不填写                               
        'addr': '',    

        # 业务私有ceph的登录名，仅当 数据集来源 方式为outer_ceph为必填字段，其余方式不填写
        'name': '',    

        # 业务私有ceph的密码，仅当 数据集来源 方式为outer_ceph为必填字段，其余方式不填写                
        'secret': ''              
      },

      # 可以查看该数据集的rtx.格式：rtx1,rtx2,rtx3. 注意：若要修改数据集，该字段可以修改
      'rtx_group': '', 

      # 备注字段. 注意：若要修改数据集，该字段可以修改
      'description_info': ''  
    },
    
    # 特别说明：model_id 和 model_params两个字段：
    # 当用户需要创建并使用新的数据集，则只需填写 model_params 字段，此时model_id 字段值应为 ''
    # 当用户使用已有的数据集，则填写 model_id 字段，使该字段不为''
    # 使用的模型集id
    'model_id': '', 
    'model_params': {
      # 模型名称，业务自定义名称
      'model_name': 'jizhi_client_demo_data_ggt',
      
      # 要上传的模型文件的本地路径  
      'path_info': './consumer/', 
      
      # 可以查看该模型的rtx,使用英文分号分割rtx名，要求格式：rtx1,rtx2,rtx3                         
      'rtx_group': '', 
      
      # 备注字段，对模型的补充说明        
      'description_info': ''         
    },
    
    'permission': {
      # 管理员组，具有对任务的操作权限，如修改任务信息、开启任务、停止任务等。使用英文逗号分割rtx名，例如yiduan,huchengliu
      'admin_group': '', 

      # 关注组，可以查看任务的运行状态、历史数据等。使用英文逗号分割rtx名，例如yiduan,huchengliu 
      'view_group': '',  
      
      # 告警组，不具有具体权限，仅仅作为接受任务告警的人员列表。使用英文逗号分割rtx名，例如yiduan,huchengliu 
      'alert_group': ''   
    }
  },

  'task_config': {
    'designated_resource': {
      # 如果host_gpu_num * host_num 没有资源,是否接收随机机型匹配，默认为False
      'strategy': {"accepted_auto_assign": False},
      
      # 申请gpu机器数 
      'host_num': 1,

      # 申请的单机卡数
      'host_gpu_num': 4,

      # 提示用户：如果没有足够资源，是否进入资源排队队列等待
      'is_resource_waiting': True,                              

      # 选择镜像，只有task_type为general_gpu_type才有.通用任务类型，要选择镜像
      'image_full_name': 'g-teg-ailab-seattle-cuda10.0-cudnn7.6.1-nccl2.4-hovorod-openmpi2.0-python3.7-pytorch1.3-cv2-speech-cv-nlp-zsh:update_10.18.2019'  
    },

    # hyperparameters: 训练任务的超参配置
    'hyper_parameters': {
      
    }
  }
}

###以下信息是查看相关列表信息，选择展示的相关项，用户可以根据需要自行修改###
'''
选择业务展示信息，不需要的可以从列表中去除， 默认展示：business_readable_name/business_flag/approval_status/business_module/is_admin
'''
bussiness_items = [
                    # 业务名称，方便业务自定用户辨识的业务名称
                    'business_readable_name',   
                    
                    # 业务标识
                    'business_flag',   

                    # 审批状态         
                    'approval_status',

                    # 业务三级模块          
                    #'business_module', 

                    # 如果该值为True,则说明操作者具有管理该业务的权限         
                    #'is_admin',   

                    # 部门              
                    # 'business_department',  

                    # 业务创建人    
                    # 'rtx'                       
                  ]

'''
选择数据集展示信息，不需要的可以从列表中去除， 默认展示：dataset_id/dataset_name/task_type/dataset_source/is_admin
'''
dataset_items = [
                  # 数据集id
                  'dataset_id',

                  # 数据集名称         
                  'dataset_name',

                  # 数据集的任务类型       
                  'task_type',

                  # 数据集来源          
                  #'dataset_source',

                  # 业务创建人     
                  # 'rtx',

                  # 如果该值为True,则说明操作者具有管理该数据集的权限                
                  #'is_admin',

                  # 创建时间           
                  # 'create_time',

                  # 创建数据集时用户上传的path_info信息        
                  # 'path_info',

                  # 可以查看该数据集的rtx          
                  # 'rtx_group',

                  # 备注字段，对数据集的补充说明          
                  # 'description_info',   
                ]

'''
选择模型展示信息，不需要的可以从列表中去除， 默认展示：model_id/model_name/task_type/is_admin
'''
model_items = [
                # 模型id
                'model_id',

                # 模型名称           
                'model_name', 

                # 模型任务类型        
                'task_type', 

                # 业务创建人         
                # 'rtx',

                # 如果该值为True,则说明操作者具有管理该数据集的权限                
                #'is_admin', 

                # 创建时间          
                # 'create_time', 

                # 模型文件路径       
                # 'path_info',

                # 可以查看模型的rtx          
                # 'rtx_group',

                # 备注字段，对模型的补充说明          
                # 'description_info',   
              ]

'''
选择任务展示信息，不需要的可以从列表中去除， 默认展示：task_flag/readable_name/task_type/is_admin/is_enable/running_type/is_instance_running/instance_state
'''
task_items = [
                # 任务标识
                'task_flag',             
                
                # 任务名称
                #'readable_name',         
                
                # 任务类型
                #'task_type',             
                
                # 任务负责人
                # 'rtx',                   
                
                # 如果该值为True,则说明操作者具有管理该数据集的权限
                #'is_admin',              
                
                # 创建时间
                # 'create_time',           
                
                # 代表任务是否启用
                #'is_enable',             
                
                # 任务的运行方式
                #'running_type',          
                
                # 是否有实例在运行中
                #'is_instance_running',   
                
                # 该任务的最新实例的运行状态，如果没有实例，则为空串''
                'instance_state',        

                # ps的准备状态
                # 'ps_state'               

                # 该任务的最新实例的运行结果，如果没有实例，则为空串''
                'is_success'
            ]

'''
选择具体任务的 实例 的展示信息，不需要的可以从列表中去除， 默认展示：instance_id/create_time/state/time_delay/is_success
'''
instance_items = [
                    # 实例id的hex表示
                    'instance_id',  
                    
                    # 实例的开始时间
                    'create_time',  
                    
                    # 实例的当前运行状态
                    'state',        
                    
                    # 当前状态开始时间
                    # 'time_start',   
                    
                    # 运行耗时时间
                    'time_delay',   
                    
                    # 执行结果
                    'is_success',   
                    
                    # 执行消息
                    # 'msg'           
                 ]
