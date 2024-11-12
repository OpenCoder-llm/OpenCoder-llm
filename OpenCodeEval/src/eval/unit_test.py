# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import io
import os
import sys
import time
import types
import unittest
import contextlib
import faulthandler
import platform
import signal
import tempfile
import subprocess
import multiprocessing
import numpy as np

from multiprocessing import Array, Value, Manager
from typing import Any, Dict, List, Tuple, Union

@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            # Avoid setting stdout or stderr if capture_output is True
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            # Avoid setting stdout or stderr if capture_output is True
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            print(f"Prevented attempt to kill PGID {pgid} with signal {sig}")

    def safe_system(command):
        print(f"Intercepted system command: {command}")
        if 'kill' in command or 'killall' in command:
            return 0  # Simulate successful execution without doing anything
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if 'kill' in command or 'killall' in command:
            return 0  # Simulate successful execution without doing anything
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if 'ps' in command:
            return b""  # Simulate no processes found
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        print(f"Intercepted subprocess run command: {args}")
        if 'kill' in args[0] or 'killall' in args[0]:
            return subprocess.CompletedProcess(args, 0, b'', b'')  # Simulate successful execution
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            print(f"Intercepted Popen command: {args}")
            kwargs['preexec_fn'] = os.setsid  # Start the process in a new session
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                print("Timeout expired, intercepted and returning None")
                return None, None

        def kill(self):
            print(f"Intercepted kill call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            print(f"Intercepted terminate call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        print(f"Intercepted os.popen command: {command}")
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        print(f"Intercepted exec command: {args}")

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error handling process {pid}: {e}")
        
        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    
    import os
    import time
    from datetime import datetime

    os.environ['TZ'] = 'UTC'
    time.tzset()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    
    if max_as_limit and max_data_limit and max_stack_limit:
        import resource
        
        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024
        
        resource.setrlimit(
            resource.RLIMIT_AS, (max_as_limit, max_as_limit)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (max_data_limit, max_data_limit)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import matplotlib.pyplot as plt
    plt.close('all')


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def unsafe_execute(
    code: str,
    test_code: str,
    timeout: float,
    stat,  # Value
    details,  # Array
):
    with safe_environment(), create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins
        
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit = 30720, max_data_limit = 30720, max_stack_limit = 10)
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()

                with time_limit(timeout):
                    suite.run(test_result)
            
            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat.value = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


import psutil

def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                if child.is_running():
                    os.kill(child.pid, signal.SIGKILL)
            except psutil.NoSuchProcess:
                continue
        if parent.is_running():
            os.kill(parent.pid, signal.SIGKILL)
    except psutil.NoSuchProcess:
        pass

def check_correctness(
    task_id: int,
    solution_id: int,
    solution: str,
    test: str,
    timeout: float,
) -> Tuple[str, np.ndarray]:

    result = {
        "task_id": task_id,
        "solution_id": solution_id
    }
    
    # shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            solution,
            test,
            timeout,
            stat,
            details,
        ),
    )

    p.start()
    p.join(timeout=timeout+1)

    if p.is_alive():
        terminate_process_tree(p.pid)
        stat.value = _TIMEOUT

    stat = _mapping[stat.value]
    details = dict(details)
    
    if not stat:
        stat = TIMEOUT
    if stat == PASS:
        if details:
            stat = FAIL

    result["passed"] = stat == PASS
    result["result"] = details
    result["solution"] = solution

    manager.shutdown()

    return result