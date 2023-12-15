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
.. _tutorial-relay-quick-start:

Quick Start Tutorial for Compiling Deep Learning Models
=======================================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Truman Tian <https://github.com/SiNZeRo>`_

This example shows how to build a neural network with Relay python frontend and
generates a runtime library for Nvidia GPU with TVM.
Notice that you need to build TVM with cuda and llvm enabled.
"""

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tvm_support_list.png
#      :align: center
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore

import sys

sys.path.insert(2,'dir') 
 
sys.path.insert(2,'/home/xilinx/tvm/vta/python')
sys.path.insert(2,'/home/xilinx/tvm/python')
sys.path.insert(2,'/home/xilinx')
sys.path.insert(2,'/home/xilinx/.local/lib/python3.10/site-packages')

sys.path.insert(2,'/usr/lib/python3/dist-packages')

from pynq import allocate
from pynq import Overlay


from todo import divide
from todo import count_op
from todo import find_input_arg


import numpy as np

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

env = vta.get_env()
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    remote = rpc.LocalSession()
    
    print("Bitstream loading")
    overlay = Overlay("/home/xilinx/vgg_test.bit")
    print("Bitstream loaded")
    
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()
    
    print("Bitstream loading")
    overlay = Overlay("/home/xilinx/vgg_test.bit")
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
    
    '''
    split_conf = [{"op_name": "nn.relu", "op_index": 3},{"op_name": "nn.relu", "op_index": 4},{"op_name": "add", "op_index": 1},{"op_name": "add", "op_index": 3},{"op_name": "nn.conv2d", "op_index": 13},
                    {"op_name": "nn.conv2d", "op_index": 16},{"op_name": "nn.dense", "op_index": 0},{"op_name": "nn.bias_add", "op_index": 0}]
    '''
    #split_conf = [{"op_name": "nn.conv2d", "op_index": 0},{"op_name": "nn.conv2d", "op_index": 1},{"op_name": "nn.conv2d", "op_index": 5}]
    #split_conf = [{"op_name": "nn.conv2d", "op_index": 1},{"op_name": "nn.conv2d", "op_index": 3},{"op_name": "nn.conv2d", "op_index": 5}]
    #split_conf = [{"op_name": "nn.conv2d", "op_index": 2},{"op_name": "nn.conv2d", "op_index": 5},{"op_name": "nn.conv2d", "op_index": 9}]
    #split_conf = [{"op_name": "nn.conv2d", "op_index": 2},{"op_name": "nn.conv2d", "op_index": 5},{"op_name": "nn.conv2d", "op_index": 8},{"op_name": "nn.conv2d", "op_index": 10}]
    
    split_conf = []
    with open('/home/xilinx/cut_point.txt', 'r') as file:
            for line in file:
                cut={"op_name": "nn.conv2d", "op_index": int(line)}
                split_conf.append(cut)
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


######################################################################
# Define Neural Network in Relay
# ------------------------------
# First, let's define a neural network with relay python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in Relay.
# Parameters are initialized with Xavier initializer.
# Relay also supports other model formats such as MXNet, CoreML, ONNX and
# Tensorflow.
#
# In this tutorial, we assume we will do inference on our device and
# the batch size is set to be 1. Input images are RGB color images of
# size 224 * 224. We can call the
# :py:meth:`tvm.relay.expr.TupleWrapper.astext()` to show the network
# structure.

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.vgg.get_workload(
    num_layers=16, batch_size=batch_size, image_shape=image_shape
)


expr = build_module.bind_params_by_name(mod["main"], params)

nodes, op_count, real_op_count=divide(expr)

print(expr)
'''
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(expr)
divide(expr)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
'''

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

#==================================================================

mods, input_idx = get_split_mod(mod["main"],params)


'''
if(True):
    for node1 in nodes:
        print("======================================")
        
        print("op num : ",node1.node_number)
        print("op type : ",node1.which_op)
        
        i=0
        for src in node1.src_name:
            print("src ",i," name : ",node1.src_name[i])
            print("src ",i," shape : ",node1.src_shape[i])
            if(node1.src_come_from[i]==-1):
                print("src ",i," from input")
            else:
                print("src ",i," from op ",node1.src_come_from[i])
            i=i+1
        
        print("dst name : ",node1.dst_name)
        print("dst shape : ",node1.dst_shape)
        
        print("======================================")
'''


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
    
'''
print("op need input:")
print(op_need_input)
print("variable name:")
print(var_map)
print("input port of each sub-model")
print(var_num_map)

print("first op number of each sub-model")
print(mod_ops)
print(len(mods))
print(len(var_num_map))
print(len(mod_ops))
'''
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

            '''
            pipe_config[mods[0]]["output"][0].connect(pipe_config[mods[1]]["input"]["data_n_0"])
            
            pipe_config[mods[0]]["output"][1].connect(pipe_config[mods[1]]["input"]["data_n_1"])
            
            pipe_config[mods[1]]["output"][0].connect(pipe_config[mods[2]]["input"]["data_n_2"])
            
            pipe_config[mods[1]]["output"][1].connect(pipe_config[mods[2]]["input"]["data_n_3"])
            
            
            pipe_config[mods[ 2 ]]["output"][0].connect(pipe_config["output"]["0"])
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
# Compilation
# -----------
# Next step is to compile the model using the Relay/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 3. The optimization passes include
# operator fusion, pre-computation, layout transformation and so on.
#
# :py:func:`relay.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, Relay does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
#
# We'll first compile for Nvidia GPU. Behind the scene, :py:func:`relay.build`
# first does a number of graph-level optimizations, e.g. pruning, fusing, etc.,
# then registers the operators (i.e. the nodes of the optimized graphs) to
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the high level IR
# into the lower intrinsic IR of the specified target backend, which is CUDA
# in this example. Then the machine code will be generated as the module library.

opt_level = 3
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

#####################################################################
# Run the generate library
# ------------------------
# Now we can create graph executor and run the module on Nvidia GPU.

# create random input
dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

data11=data


# create module
'''
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input("data", data)
# run
start1 = time.time()
module.run()
end1 =  time.time()
print("Execution time：%f" % (end1 - start1))
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print(out.flatten()[0:10])

######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.

####################################################

# save the graph, lib and params into separate files
from tvm.contrib import utils

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
print(temp.listdir())

####################################################

# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
start1 = time.time()
module.run(data=input_data)
end1 =  time.time()
print("Execution time：%f" % (end1 - start1))
out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)
'''
normal_outputs = []

iterations=1

start12 = time.time()

for i in range(iterations):

    pipeline_module_test.set_input("data_a", tvm.nd.array(data11))

    start1 = time.time()
    #print("start timer")
    
    
    pipeline_module_test.run()
    end1 =  time.time()
    
    print("Execution time of pipeline：%f" % (end1 - start1))
    
end12 =  time.time()
print("Execution time：%f" % (end12 - start12))
    
for i in range(iterations):
    statistic_time = 0
    outputs = pipeline_module_test.get_output()
                
    while len(outputs) == 0:
        outputs = pipeline_module_test.get_output()
        statistic_time = statistic_time + 1
        # Setting the timeout to 10 seconds.
        #assert statistic_time < 5
        time.sleep(0.01)
                
    print("-----------------------------")
                
    for i in range(len(outputs)):
                    #tvm.testing.assert_allclose(normal_outputs[k][i], outputs[i].numpy())
                    
        print(outputs[i].numpy().flatten()[0:10])
        aa=i
                    #print(normal_outputs[k][i])

                    #assert not (normal_output[i] == wrong_output[i]).all()

                    #assert pipeline_module_test.num_executing_pipeline == round + 1
    print("-----------------------------")

end12 =  time.time()
print("Execution time：%f" % (end12 - start12))
            # Reset the cpu affinity after a test.
reset_cpu_affinity(affinity)