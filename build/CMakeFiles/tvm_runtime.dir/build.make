# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xilinx/tvm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xilinx/tvm/build

# Include any dependencies generated for this target.
include CMakeFiles/tvm_runtime.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tvm_runtime.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tvm_runtime.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tvm_runtime.dir/flags.make

# Object files for target tvm_runtime
tvm_runtime_OBJECTS =

# External object files for target tvm_runtime
tvm_runtime_EXTERNAL_OBJECTS = \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/builtin_fp16.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/c_runtime_api.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/const_loader_module.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/container.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/cpu_device_api.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/debug.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/dso_library.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/file_utils.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/library_module.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/logging.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/metadata.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/minrpc/minrpc_logger.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/module.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/name_transforms.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/ndarray.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/object.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/packed_func.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/profiling.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/registry.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/source_utils.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/static_library.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/system_library.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/thread_pool.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/threading_backend.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/bytecode.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/executable.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/memory_manager.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/vm.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/workspace_pool.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_channel.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_device_api.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_endpoint.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_event_impl.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_local_session.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_module.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_pipe_impl.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_server_env.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_session.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_socket_impl.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor_factory.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/debug/graph_executor_debug.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/profiler/vm.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/aot_executor/aot_executor.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/aot_executor/aot_executor_factory.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/pipeline/pipeline_executor.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/pipeline/pipeline_scheduler.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/random/random.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/sort/sort.cc.o" \
"/home/xilinx/tvm/build/CMakeFiles/tvm_libinfo_objs.dir/src/support/libinfo.cc.o"

libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/builtin_fp16.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/c_runtime_api.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/const_loader_module.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/container.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/cpu_device_api.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/debug.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/dso_library.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/file_utils.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/library_module.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/logging.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/metadata.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/minrpc/minrpc_logger.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/module.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/name_transforms.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/ndarray.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/object.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/packed_func.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/profiling.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/registry.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/source_utils.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/static_library.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/system_library.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/thread_pool.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/threading_backend.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/bytecode.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/executable.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/memory_manager.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/vm.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/workspace_pool.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_channel.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_device_api.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_endpoint.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_event_impl.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_local_session.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_module.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_pipe_impl.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_server_env.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_session.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_socket_impl.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor_factory.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/debug/graph_executor_debug.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/profiler/vm.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/aot_executor/aot_executor.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/aot_executor/aot_executor_factory.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/pipeline/pipeline_executor.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/pipeline/pipeline_scheduler.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/random/random.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/sort/sort.cc.o
libtvm_runtime.so: CMakeFiles/tvm_libinfo_objs.dir/src/support/libinfo.cc.o
libtvm_runtime.so: CMakeFiles/tvm_runtime.dir/build.make
libtvm_runtime.so: libbacktrace/lib/libbacktrace.a
libtvm_runtime.so: CMakeFiles/tvm_runtime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xilinx/tvm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX shared library libtvm_runtime.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tvm_runtime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tvm_runtime.dir/build: libtvm_runtime.so
.PHONY : CMakeFiles/tvm_runtime.dir/build

CMakeFiles/tvm_runtime.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tvm_runtime.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tvm_runtime.dir/clean

CMakeFiles/tvm_runtime.dir/depend:
	cd /home/xilinx/tvm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xilinx/tvm /home/xilinx/tvm /home/xilinx/tvm/build /home/xilinx/tvm/build /home/xilinx/tvm/build/CMakeFiles/tvm_runtime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tvm_runtime.dir/depend

