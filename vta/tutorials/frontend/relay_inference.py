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
sys.path.insert(2,'/home/xilinx/.local/lib/python3.7/site-packages')

from pynq import allocate
from pynq import Overlay

from todo import divide
from todo import count_op
from todo import count_all_op
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
    overlay = Overlay("/home/xilinx/vgg11.bit")
    print("Bitstream loaded")
    
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()
    
    print("Bitstream loading")
    overlay = Overlay("/home/xilinx/vgg11.bit")
    print("Bitstream loaded")
    
    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")


# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)



#####################################################################
# Run the generate library
# ------------------------
# Now we can create graph executor and run the module on Nvidia GPU.

# create random input

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)



target = "llvm"
dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

#print(data)

data11=data

'''
# create module

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
'''
######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.

####################################################

# save the graph, lib and params into separate files
from tvm.contrib import utils

#temp = utils.tempdir()
#path_lib = temp.relpath("deploy_lib.tar")]
path_lib = "/home/xilinx/deploy_lib.tar"
#lib.export_library(path_lib)
#print(temp.listdir())
print(path_lib)

####################################################

# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))

start12 = time.time()

module.run(data=input_data)

end12 =  time.time()
print("Execution time：%f" % (end12 - start12))

out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
#tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)
'''
normal_outputs = []

for i in range(3):

    pipeline_module_test.set_input("data_a", tvm.nd.array(data11))

    start1 = time.time()
    print("start timer")
    pipeline_module_test.run()
    end1 =  time.time()
    print("Execution time of pipeline：%f" % (end1 - start1))
    
for i in range(3):
    statistic_time = 0
    outputs = pipeline_module_test.get_output()
                
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
reset_cpu_affinity(affinity)
'''