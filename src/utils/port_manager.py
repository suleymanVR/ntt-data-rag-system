"""
Port management utilities for avoiding port conflicts.
"""

import socket
import subprocess
import logging
from typing import List, Any

logger = logging.getLogger(__name__)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is currently in use.
    
    Args:
        port: Port number to check
        host: Host address to check
        
    Returns:
        True if port is in use, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.warning(f"Error checking port {port}: {e}")
        return True  # Assume port is in use if we can't check


def find_free_port(start_port: int = 8000, max_attempts: int = 50) -> int:
    """
    Find a free port starting from start_port.
    
    Args:
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        Free port number
        
    Raises:
        RuntimeError: If no free port found
    """
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            logger.info(f"Found free port: {port}")
            return port
    
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


def kill_process_on_port(port: int) -> bool:
    """
    Kill process using specified port (Windows).
    
    Args:
        port: Port number
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find process using the port
        result = subprocess.run(
            ["netstat", "-ano"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        pids = []
        
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.append(int(pid))
        
        # Kill processes
        for pid in set(pids):  # Remove duplicates
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                             capture_output=True, check=True)
                logger.info(f"Killed process {pid} using port {port}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to kill process {pid}: {e}")
                return False
                
        return len(pids) > 0
        
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")
        return False


def get_available_port(preferred_port: int = 8000, auto_kill: bool = False) -> int:
    """
    Get an available port, with option to kill existing process.
    
    Args:
        preferred_port: Preferred port number
        auto_kill: Whether to automatically kill process on preferred port
        
    Returns:
        Available port number
    """
    if not is_port_in_use(preferred_port):
        logger.info(f"Port {preferred_port} is available")
        return preferred_port
    
    logger.warning(f"Port {preferred_port} is in use")
    
    if auto_kill:
        logger.info(f"Attempting to free port {preferred_port}")
        if kill_process_on_port(preferred_port):
            # Wait a moment and check again
            import time
            time.sleep(1)
            if not is_port_in_use(preferred_port):
                logger.info(f"Successfully freed port {preferred_port}")
                return preferred_port
    
    # Find alternative port
    alternative_port = find_free_port(preferred_port + 1)
    logger.info(f"Using alternative port {alternative_port}")
    return alternative_port


def get_process_info_on_port(port: int) -> List[dict[str, Any]]:
    """
    Get information about processes using specified port.
    
    Args:
        port: Port number
        
    Returns:
        List of process information dictionaries
    """
    processes = []
    
    try:
        result = subprocess.run(
            ["netstat", "-ano"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        try:
                            # Get process name
                            proc_result = subprocess.run(
                                ["tasklist", "/FI", f"PID eq {pid}"], 
                                capture_output=True, 
                                text=True,
                                check=True
                            )
                            proc_lines = proc_result.stdout.split('\n')
                            process_name = "Unknown"
                            for proc_line in proc_lines:
                                if pid in proc_line:
                                    process_name = proc_line.split()[0]
                                    break
                            
                            processes.append({
                                "pid": int(pid),
                                "name": process_name,
                                "address": parts[1] if len(parts) > 1 else "",
                                "state": parts[3] if len(parts) > 3 else ""
                            })
                        except Exception:
                            processes.append({
                                "pid": int(pid),
                                "name": "Unknown",
                                "address": parts[1] if len(parts) > 1 else "",
                                "state": parts[3] if len(parts) > 3 else ""
                            })
    
    except Exception as e:
        logger.error(f"Error getting process info for port {port}: {e}")
    
    return processes
