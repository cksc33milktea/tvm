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

from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform

conv_count=0

class node:

    def __init__(self, node_number):
        self.node_number=node_number
        

def divide(mod):
    
    #print(mod)
    
    
    anf = run_opt_pass(mod, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    
    #print(anf)
    
    num=0
    
    num_real_op=0
    
    loop=anf
    
    nodes=[]
    varibale_map=[]
    
    
    
    while(hasattr(loop,'body')):
    
        flag=0
        
        src_name=[]
        src_shape=[]
        src_come_from=[]
        
        dst_shape=0
        
        if(hasattr(loop.body,'value')):
            if(hasattr(loop.body.value,'op')):
                flag=1
                num=num+1
                num_real_op=num_real_op+1
            '''
            elif(hasattr(loop.body.value,'tuple_value')):
                flag=2
                num=num+1
            '''
        
        
        #print("------------------------------")
        
        #print(loop.body)
        #print("op ",num)
        
        
        #print("+++++++++++++++++++++++++++++++++")
        if(flag==1):
            if(hasattr(loop.body,'var')):
                #print(loop.body.var)
                #print(loop.body.var.name_hint)
                dst_name=loop.body.var.name_hint
                
                if(hasattr(loop.body.var.checked_type,'concrete_shape')):
                    dst_shape=loop.body.var.checked_type.concrete_shape
                    '''
                    for shape in loop.body.var.checked_type.concrete_shape:
                        print(shape)
                    '''
                    
                #print("+++++++++++++++++++++++++++++++++")
            if(hasattr(loop.body,'value')):
                if(hasattr(loop.body.value,'op')):
                    which_op=loop.body.value.op
                    #num=num+1
                if(hasattr(loop.body.value,'args')):
                    for arg in loop.body.value.args:
                        src_name.append(arg.name_hint)
                        if(hasattr(arg.checked_type,'concrete_shape')):
                            src_shape.append(arg.checked_type.concrete_shape)
                        else:
                            src_shape.append([])
                        '''
                        for shape in arg.checked_type.concrete_shape:
                            print(shape)
                        '''
            #print("------------------------------")
        
            
            current_node=node(num-1)
            current_node.which_op=which_op.name
            current_node.src_shape=src_shape
            current_node.src_name=src_name
            current_node.dst_shape=dst_shape
            current_node.dst_name=dst_name
            nodes.append(current_node)
            varibale_map.append(dst_name)
            
            if(current_node.which_op=="nn.batch_norm"):
                current_node.can_cut=False
            else:
                current_node.can_cut=True
            
            
            for name in src_name:
                i=0
                for source in varibale_map:
                    if(source==name):
                        break
                    i=i+1
                if(i==len(varibale_map)):
                    src_come_from.append(-1)
                else:
                    src_come_from.append(i)
                
            current_node.src_come_from=src_come_from
        
        elif(flag==2):
        
            which_op="Assign"
            
            if(hasattr(loop.body,'var')):
                dst_name=loop.body.var.name_hint
                
                if(hasattr(loop.body.var.checked_type,'concrete_shape')):
                    dst_shape=loop.body.var.checked_type.concrete_shape
                    #print(dst_shape)
                    
                #print("+++++++++++++++++++++++++++++++++")
            if(hasattr(loop.body,'value')):
            
                if(hasattr(loop.body.value,'tuple_value')):
                    #print(dir(loop.body.value.tuple_value))
                    src_name.append(loop.body.value.tuple_value.name_hint)
                    src_shape.append(0)
                    
            #print("------------------------------")
        
            
            current_node=node(num-1)
            current_node.which_op=which_op
            current_node.src_shape=src_shape
            current_node.src_name=src_name
            current_node.dst_shape=dst_shape
            current_node.dst_name=dst_name
            nodes.append(current_node)
            varibale_map.append(dst_name)
            
            
            
            for name in src_name:
                i=0
                for source in varibale_map:
                    if(source==name):
                        break
                    i=i+1
                if(i==len(varibale_map)):
                    src_come_from.append(-1)
                else:
                    src_come_from.append(i)
                
            current_node.src_come_from=src_come_from
        
        loop=loop.body
        
    
    global conv_count
    
    for node1 in nodes:
        print("======================================")
        
        print("op num : ",node1.node_number)
        print("op type : ",node1.which_op)
        print("op type : ",node1.can_cut)
        
        if(node1.which_op=="nn.conv2d"):
            conv_count=conv_count+1
        
        
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
    
    
    path = 'conv.txt'
    f = open(path, 'w')
    
    f.write(str(conv_count))
    f.write('\n')
    
    for node1 in nodes:
        
        if(node1.which_op=="nn.conv2d"):
        
            tup=node1.src_shape[0]
            f.write(str(tup[1]))
            f.write('\n')
            f.write(str(tup[2]))
            f.write('\n')
            f.write(str(tup[3]))
            f.write('\n')
            
            tup=node1.dst_shape
            f.write(str(tup[1]))
            f.write('\n')
            f.write(str(tup[2]))
            f.write('\n')
            f.write(str(tup[3]))
            f.write('\n')
        
        #f.write(str(node1.dst_shape))
    
    f.close()
    
    
    print("!!!!!!!!!!!!!!!!!1")
    
    return nodes, num, num_real_op
    
def count_op(mod):
    
    #print(mod)
    
    
    anf = run_opt_pass(mod, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    
    #print(anf)
    
    num=0
    
    loop=anf
    
    nodes=[]
    varibale_map=[]
    
    
    
    while(hasattr(loop,'body')):
    
        if(hasattr(loop.body,'value')):
            if(hasattr(loop.body.value,'op')):
                flag=1
                num=num+1
            '''
            elif(hasattr(loop.body.value,'tuple_value')):
                flag=2
                num=num+1
            '''
        loop=loop.body

    
    return num
    
def find_input_arg(mod):
    '''
    print("-----------------------------------------")
    print(mod)
    print("-----------------------------------------")
    '''
    anf = run_opt_pass(mod, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    
    args=[]
    for arg in anf.params:
        args.insert(0,arg.name_hint)

    #print(args)
    return args
    
def count_all_op(mod):
    
    #print(mod)
    
    
    anf = run_opt_pass(mod, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    
    #print(anf)
    
    num=0
    
    loop=anf
    
    nodes=[]
    varibale_map=[]
    
    
    
    while(hasattr(loop,'body')):
    
        num=num+1
        loop=loop.body

    
    return num

