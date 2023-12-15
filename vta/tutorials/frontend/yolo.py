# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile YOLO-V2 and YOLO-V3 in DarkNet Models
=============================================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This article is an introductory tutorial to deploy darknet models with TVM.
All the required models and libraries will be downloaded from the internet by the script.
This script runs the YOLO-V2 and YOLO-V3 Model with the bounding boxes
Darknet parsing have dependancy with CFFI and CV2 library
Please install CFFI and CV2 before executing this script

.. code-block:: bash

  %%shell
  pip install cffi opencv-python

"""
import sys

sys.path.insert(2,'dir') 
 
sys.path.insert(2,'/home/xilinx/tvm/vta/python')
sys.path.insert(2,'/home/xilinx/tvm/python')
sys.path.insert(2,'/home/xilinx')
sys.path.insert(2,'/home/xilinx/.local/lib/python3.10/site-packages')

sys.path.insert(2,'/usr/lib/python3/dist-packages')
#sys.path.insert(2,'/usr/local/share/pynq-venv/lib/python3.10/site-packages')
from pynq import allocate
from pynq import Overlay



from todo import divide
from todo import count_op
from todo import count_all_op
from todo import find_input_arg

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

import time

import pytest
import os
import time
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform, build_module
from tvm.relay.testing import run_opt_pass
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build
from tvm._ffi import get_global_func
from tvm.contrib import cc as _cc
import os,time

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack

from tvm.contrib.pipeline_executor import PipelineExecutorFactoryModule

# numpy and matplotlib
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

# tvm, relay
import tvm
from tvm import te
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet


env = vta.get_env()
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    remote = rpc.LocalSession()
    
    print("Bitstream loading")
    overlay = Overlay("/home/xilinx/thesis_poor/yolo.bit")
    print("Bitstream loaded")
    
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()
    
    print("Bitstream loading")
    overlay = Overlay("/home/xilinx/thesis_poor/yolo.bit")
    print("Bitstream loaded")
    
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")


# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)


var_map=[]
var_num_map=[]
mod_ops=[]
op_need_input=[]

real_op_count=0
stage_count=0

def reset_cpu_affinity(affinity):
    # Restore the CPU affinity into the default value.
    config_threadpool = get_global_func("runtime.config_threadpool")
    config_threadpool(-2, 0)
    os.sched_setaffinity(0, affinity)
    
    
def graph_split(expr, split_conf, params=None):
    """Splitting the graph into a list of subgraphs"""
    
    
    def get_dep_var(sub_var_dep):
        return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

    def parse_dependency(value, snode_dep, new_input_idx, stage_count):
        new_args = []
        need_update = False
        #global var_map
        for var in value.args:
            is_free_var = False
            for dep in snode_dep[:-1]:
                if var in dep["nodes"]:
                    # Mark the previous subgraph node as a dependency.
                    dep["nodes"][var] += 1
                    dep["ref_nodes"][var] = dep["nodes"][var]
                    # The var of this call is a free_var
                    is_free_var = True
            # if the var of this call is a free_var, recreate it and give it a fixed input name.
            if is_free_var:
                need_update = True
                new_args.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
                global op_need_input
                op_need_input.append(stage_count)
                #print("stage count: ",stage_count)
                #var_map.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
                new_input_idx += 1
            else:
                new_args.append(var)
                #var_map.append(var)
        # if the 'tvm.relay.expr.Call' has a free_var, recreate it with new name as 'data_n_*'.
        if need_update:
            value = tvm.relay.expr.Call(
                value.op, new_args, value.attrs, value.type_args, value.span
            )
        return value, snode_dep, new_input_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, pipeline_mods, split_conf, constant_expr):
        # Enumurate all operators of compute graph, then split the compute graph into a group of
        # subgraph.
        
        
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal snode_dep
        cur_node_dep = snode_dep[len(snode_dep) - 1]
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # record the constant expr to make sure all sugraphs can find correct constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
            if isinstance(value, tvm.relay.expr.Call):
                new_args = []
                # build current var list
                cur_node_dep["nodes"][anf.var] = 0
                # Get the dependency information of the nodes.
                
                global stage_count
                    
                
                
                value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx, stage_count)
                
                stage_count=stage_count+1
                
                
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                    split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                    # if a operator name and repeating count in the network match with the values
                    # of the 'split configuration', then this place is where we should do the
                    # graph splitting.
                    
                    
                    
                    #print(real_op_count)

                    
                    if (
                        
                        split_conf
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        split_conf.pop(0)
                    
                    #if(stage_count<real_op_count-1):
                        # Do graph splitting.
                        #split_conf.pop(0)
                        
                        #print(real_op_count)
                        #print(stage_count)
                        
                        #stage_count=stage_count+1
                        
                        global var_map
                        global var_num_map
                        
                        snode_dep.append({"nodes": {}, "ref_nodes": {}})
                        ann = _recursion(
                            anf.body,
                            pipeline_mods,
                            split_conf,
                            constant_expr,
                        )
                        snode_dep.pop()
                        dep_vars = get_dep_var(snode_dep)
                        # When the nodes of the current subgraph are the depedency node of another
                        # subgraph, we need to set them as the output of current subgraph.
                        body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                        # when the operator of current subgraph uses previous subgraph constant
                        # as the argument of a "relay.expr.call", such constant may become a free
                        # varaible if the constant does not exist in the current subgraph.
                        # merge the previous constant with current subgraph to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)
                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        
                        args=find_input_arg(mod["main"])
                        
                        #print(args)
                        var_num=0
                        
                        
                        '''
                        if(hasattr(ann,'args')):
                            for arg in ann.args:
                                if(hasattr(arg,'name_hint')):
                                    #print(arg.name_hint)
                                    var_map.insert(0,arg.name_hint)
                                    var_num=var_num+1
                                elif(hasattr(arg,'tuple_value')):
                                    #print(arg.tuple_value.name_hint)
                                    var_map.insert(0,arg.tuple_value.name_hint)
                                    var_num=var_num+1
                        elif(hasattr(ann,'name_hint')):
                            #print(ann.name_hint)
                            var_map.insert(0,ann.name_hint)
                            var_num=var_num+1
                           ''' 
                        
                        for arg in args:
                            var_map.insert(0,arg)
                        
                        var_num_map.insert(0,len(args))
                        # Return the last node of the current subgraph.
                        return tvm.relay.expr.Let(anf.var, value, body)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
            )
        else:
            return anf

    snode_dep = [{"nodes": {}, "ref_nodes": {}}]
    pipeline_mods = []
    operator_index_map = {}
    # Used to tracking new input which caused by graph splitting.
    new_input_idx = 0
    constant_expr = None
    subgraph_split_conf = split_conf.copy()
    # Binding the parameters.
    if params:
        expr = build_module.bind_params_by_name(expr, params)
    
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    
    ann = _recursion(
        anf,
        pipeline_mods,
        subgraph_split_conf,
        constant_expr,
    )
    
    '''
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for modd in pipeline_mods:
        print("--------------------------------")
        print(modd)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    '''
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    #print(mod)
    return pipeline_mods, new_input_idx
    
    
def get_split_mod(mod,params):
    #mod, dshape = get_network()
    
    
    #nodes=divide(mod)
    
    '''
    
    for node in nodes:
        print("======================================")
        
        print("op num : ",node.node_number)
        print("op type : ",node.which_op)
        
        i=0
        for src in node.src_name:
            print("src ",i," name : ",node.src_name[i])
            print("src ",i," shape : ",node.src_shape[i])
            if(node.src_come_from[i]==-1):
                print("src ",i," from input")
            else:
                print("src ",i," from op ",node.src_come_from[i])
            i=i+1
        
        print("dst name : ",node.dst_name)
        print("dst shape : ",node.dst_shape)
        
        print("======================================")
    
    '''
    
    
    #print(mod)
    
    split_conf = [{"op_name": "nn.relu", "op_index": 3},{"op_name": "nn.relu", "op_index": 4},{"op_name": "add", "op_index": 1},{"op_name": "add", "op_index": 3},{"op_name": "nn.conv2d", "op_index": 13},
                    {"op_name": "nn.conv2d", "op_index": 16},{"op_name": "nn.dense", "op_index": 0},{"op_name": "nn.bias_add", "op_index": 0}]
    
    #split_conf = [{"op_name": "nn.relu", "op_index": 3}]
    
    split_conf = [{"op_name": "nn.conv2d", "op_index": 4},{"op_name": "nn.conv2d", "op_index": 9},{"op_name": "nn.conv2d", "op_index": 14},{"op_name": "nn.conv2d", "op_index": 19}]
    #split_conf = []
    
    
    mods, input_idx = graph_split(mod, split_conf,params)
    
    return mods, input_idx
    
    
def recreate_parameters(mod):
    # Get the binding parameters from a module, then create the same parameters with different data.
    # This function is used to test the "parameter" connection.
    
    with vta.build_config(opt_level=0):
    #with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, "llvm")
    '''
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print(lib.ir_mod)
    print(lib)
    print(lib.lib)
    print(dir(mod))
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    '''
    mod_customized_params = {}
    for key, value in lib.params.items():
        
        new_value = value.numpy() + np.full(value.shape, 0).astype(value.dtype)
        mod_customized_params[key] = tvm.nd.array(new_value)
    return mod_customized_params, mod





print("!!!!!!!!!!!!!!!!!!!!")
######################################################################
# Choose the model
# -----------------------
# Models are: 'yolov2', 'yolov3' or 'yolov3-tiny'

# Model name
MODEL_NAME = "yolov2"

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + ".cfg"
WEIGHTS_NAME = MODEL_NAME + ".weights"
REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"
CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME
print("!!!!!!!!!!!!!!!!!!!!")
cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
print("!!!!!!!!!!!!!!!!!!!!")
# Download and Load darknet library
if sys.platform in ["linux", "linux2"]:
    DARKNET_LIB = "libdarknet2.0.so"
    DARKNET_URL = REPO_URL + "lib/" + DARKNET_LIB + "?raw=true"
elif sys.platform == "darwin":
    DARKNET_LIB = "libdarknet_mac2.0.so"
    DARKNET_URL = REPO_URL + "lib_osx/" + DARKNET_LIB + "?raw=true"
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)
print("!!!!!!!!!!!!!!!!!!!!")
lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
print(lib_path)
lib_path='/home/xilinx/darknet/libdarknet.so'
print(lib_path)

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1
print("!!!!!!!!!!!!!!!!!!!!")
'''
net.c=3
net.h=224
net.w=224
'''

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {"data": data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
print("!!!!!!!!!!!!!!!!!!!!")



expr = build_module.bind_params_by_name(mod["main"], params)

nodes, op_count, real_op_count=divide(expr)

print(expr)

co=count_all_op(expr)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
print(co)


mods, input_idx = get_split_mod(mod["main"],params)


i=0
total=0
for mod_pipeline in mods:
    #print("-----------------------------------")
    #print(mod_pipeline)
    num=count_op(mod_pipeline["main"])
    #print(num)
    #print("-----------------------------------")
    i=i+1
    total=total+num
    mod_ops.append(total)


print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("start connection")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

output_now=[]
op_input_parameter_count=[]

for mod in mods:
    print(mod)
    output_now.append(0)

for i in range(mod_ops[len(mod_ops)-1]):
    op_input_parameter_count.append(0)

affinity = os.sched_getaffinity(0)

pipe_config = pipeline_executor_build.PipelineConfig()

customized_parameters, customized_parameters_mod = recreate_parameters(mods[0])


if(True):
    if(True):
        if(True):
        
            # The global parameters group named "param_0" will be connected to "mods[0]" as parameters.
            pipe_config["param_group"]["param_0"].connect(pipe_config[mods[0]]["param"])
            
            
            pipe_config["input"]["data_a"].connect(pipe_config[mods[0]]["input"]["data"])

            

            
            pipe_config[mods[0]]["output"][0].connect(pipe_config[mods[1]]["input"]["data_n_0"])
            
            #pipe_config[mods[0]]["output"][1].connect(pipe_config[mods[1]]["input"]["data_n_1"])
            
            pipe_config[mods[1]]["output"][0].connect(pipe_config[mods[2]]["input"]["data_n_1"])
            
            pipe_config[mods[2]]["output"][0].connect(pipe_config[mods[3]]["input"]["data_n_2"])
            
            pipe_config[mods[3]]["output"][0].connect(pipe_config[mods[4]]["input"]["data_n_3"])
            
            '''
            i=0
            var_position=0
            mod_first_op=0
            
            for mod_pipeline in mods:
                #print("mod_first_op:",mod_first_op)
                if(i==0):
                    #print("-------------------------------------------")
                    #print(mod_pipeline)
                    mod_first_op=mod_ops[i]
                    i=i+1
                    continue
                
                #print("-------------------------------------------")
                #print(mod_pipeline)
                port_num=var_num_map[i-1]
                #print(port_num)
                #print(mod_now)
                #print(len(nodes[mod_now].src_come_from))
                #print("++++++")
                
                
                for j in range(port_num):
                    #print(var_map[var_position+port_num-j-1])
                    
                    op_now=op_need_input[0]
                    op_need_input.pop(0)
                    
                    while(True):
                        #print(op_now,nodes[op_now].which_op,op_input_parameter_count[op_now],nodes[op_now].src_come_from[op_input_parameter_count[op_now]],mod_first_op)
                        if(nodes[op_now].src_come_from[op_input_parameter_count[op_now]]==-1):
                            op_input_parameter_count[op_now]=op_input_parameter_count[op_now]+1
                        elif (nodes[op_now].src_come_from[op_input_parameter_count[op_now]]>=mod_first_op):
                            op_input_parameter_count[op_now]=op_input_parameter_count[op_now]+1
                        else:
                            break
                    
                    idx=0
                    while(True):
                        #print(mod_ops[idx],op_now,nodes[op_now].src_come_from[op_input_parameter_count[op_now]])
                        if(mod_ops[idx]>nodes[op_now].src_come_from[op_input_parameter_count[op_now]]):
                            break
                        #print(idx)
                        idx=idx+1
                    #print(nodes[op_now].src_come_from[0])
                    print(idx,output_now[ idx ],i,var_map[var_position+j])
                    pipe_config[mods[  idx  ]]["output"][output_now[ idx ]].connect(pipe_config[mods[i]]["input"][var_map[var_position+j]])
                    output_now[ idx ]=output_now[ idx ]+1
                    op_input_parameter_count[op_now]=op_input_parameter_count[op_now]+1
                    
                var_position=var_position+port_num
                
                mod_first_op=mod_ops[i]
                i=i+1
    
            '''
            
            pipe_config[mods[ len(mods)-1 ]]["output"][0].connect(pipe_config["output"]["0"])
            
            '''
            pipe_config[mods[0]].target = "llvm"
            pipe_config[mods[0]].dev = tvm.cpu(0)
            pipe_config[mods[0]].cpu_affinity = "0"
            
            pipe_config[mods[1]].target = "llvm"
            pipe_config[mods[1]].dev = tvm.cpu(0)
            pipe_config[mods[1]].cpu_affinity = "1"
            '''
            
            
            aff=0
            for mod in mods:
                pipe_config[mod].target = "llvm"
                pipe_config[mod].dev = tvm.cpu(0)
                #pipe_config[mod].cpu_affinity = str(aff)
                aff=aff+1
            
            mconfig = pipe_config.get_config()
            
            
            with vta.build_config(opt_level=0):
            #with tvm.transform.PassContext(opt_level=3):
                pipe_configs=pipe_config
                libs = {}
                config = pipe_configs.get_config()
                if "module_connection" not in config:
                    raise RuntimeError('"module_connection" is missing')
                if "input_connection" not in config:
                    raise RuntimeError('"input_connection" is missing')
                if "param_connection" not in config:
                    raise RuntimeError('"param_connection" is missing')

                mod_n_configs = config["module_connection"]
                config_len = len(mod_n_configs)
                module_string_config = [{} for _ in range(config_len)]
                # Use hardware configurations to build backend modules for each subgraph.
                for ir_mod, mod_config in mod_n_configs.items():
                    pipe_config = mod_config["pipeline"].copy()
                    mod_idx = pipe_config["mod_idx"]
                    dev = mod_config["dev"]
                    target = mod_config["target"]
                    
                    # Callers may need to use a customized building function to wrap the pre-building logic
                    # and the backend building logic. For example, in order to support a backend which only
                    # can do "int8" computation, the caller may need to merge the "quantization" logic
                    # into the building logic to creat a customized building function.

                    lib = relay.build(
                        ir_mod,
                        target,
                        params=mod_config["params"],
                        target_host=mod_config["target_host"],
                        mod_name=mod_config["mod_name"],
                    )
                    

                    pipe_config["dev"] = "{},{}".format(dev.device_type, dev.device_id)
                    # Use "mod_idx" as the key to create a "module_connection" map which is not only
                    # for the module index but also for the module connection used to build the pipeline.
                    module_string_config[mod_idx] = pipe_config
                    libs[mod_idx] = {
                        "lib": lib,
                        "dev": dev,
                        "fcompile": mod_config["fcompile"],
                        #"export_cc": mod_config["export_cc"],
                    }

                # Creating a text form configuration to record the "input_connection" and the
                # "module_connection" information. The "input_connection" is used to record the
                # map of global input and subgraph input, and the "module_connection" is used to
                # record module dependency.
                string_config = {}
                string_config["param_connection"] = config["param_connection"]
                string_config["input_connection"] = config["input_connection"]
                string_config["module_connection"] = module_string_config
                
                pipeline_mod_factory=PipelineExecutorFactoryModule(libs, string_config)
            
            pipeline_module_test = pipeline_executor.PipelineModule(pipeline_mod_factory)
            
            pipeline_module_test.set_params("param_0", customized_parameters)
            
            
            

######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {"data": data.shape}


print("Compiling the model...")
'''
with vta.build_config(opt_level=0):
#with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
'''


[neth, netw] = shape["data"][2:]  # Current image shape is 608x608
######################################################################
# Load a test image
# -----------------
test_image = "dog.jpg"
print("Loading the test image...")
img_url = REPO_URL + "data/" + test_image + "?raw=true"
img_path = download_testdata(img_url, test_image, "data")

data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.

'''
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))

# set inputs
m.set_input("data", tvm.nd.array(data.astype(dtype)))
# execute
print("Running the test image...")

# detection
# thresholds
thresh = 0.5
nms_thresh = 0.45


start12 = time.time()

m.run()

end12 =  time.time()
print("Execution time：%f" % (end12 - start12))
'''

thresh = 0.5
nms_thresh = 0.45

normal_outputs = []

iterations=1

start12 = time.time()

for i in range(iterations):

    pipeline_module_test.set_input("data_a", tvm.nd.array(data.astype(dtype)))
    #print(tvm.nd.array(data.astype(dtype)))

    start1 = time.time()
    print("start timer")
    pipeline_module_test.run()
    end1 =  time.time()
    print("Execution time of pipeline：%f" % (end1 - start1))

for i in range(iterations):
    statistic_time = 0
    outputs = pipeline_module_test.get_output()
    #print(outputs)
    #print(outputs[i].numpy().flatten())
                
    while len(outputs) == 0:
        outputs = pipeline_module_test.get_output()
        statistic_time = statistic_time + 1
        # Setting the timeout to 10 seconds.
        assert statistic_time < 5
        time.sleep(1)
                
    print("-----------------------------")
                
    for i in range(len(outputs)):
                    #tvm.testing.assert_allclose(normal_outputs[k][i], outputs[i].numpy())
                    
        print(outputs[i].numpy().flatten()[0:10])
        aa=i
                    #print(normal_outputs[k][i])

                    #assert not (normal_output[i] == wrong_output[i]).all()

                    #assert pipeline_module_test.num_executing_pipeline == round + 1
    print("-----------------------------")


            # Reset the cpu affinity after a test.
            
end12 =  time.time()
print("Execution time：%f" % (end12 - start12))
reset_cpu_affinity(affinity)

# get outputs

tvm_out = []
if MODEL_NAME == "yolov2":
    layer_out = {}
    layer_out["type"] = "Region"
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    #layer_attr = pipeline_module_test.get_output(2).numpy()
    layer_attr=np.array([5, 425, 13, 13, 80, 4, 0], dtype = int)  
    
    #layer_out["biases"] = pipeline_module_test.get_output(1).numpy()
    layer_out["biases"]=np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
    
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
    
    #outputs = pipeline_module_test.get_output(0)
    
    layer_out["output"] = outputs[0].numpy().reshape(out_shape)
    layer_out["classes"] = layer_attr[4]
    layer_out["coords"] = layer_attr[5]
    layer_out["background"] = layer_attr[6]
    tvm_out.append(layer_out)


# do the detection and bring up the bounding boxes
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
    (netw, neth), (im_w, im_h), thresh, 1, tvm_out
)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = "coco.names"
coco_url = REPO_URL + "data/" + coco_name + "?raw=true"
font_name = "arial.ttf"
font_url = REPO_URL + "data/" + font_name + "?raw=true"
coco_path = download_testdata(coco_url, coco_name, module="data")
font_path = download_testdata(font_url, font_name, module="data")

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

tvm.relay.testing.yolo_detection.show_detections(img, dets, thresh, names, last_layer.classes)
tvm.relay.testing.yolo_detection.draw_detections(
    font_path, img, dets, thresh, names, last_layer.classes
)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
