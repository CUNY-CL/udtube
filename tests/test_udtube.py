from udtube_package.UDTube.cli import udtube_python_interface

if __name__ == "__main__":
    udtube_python_interface(["fit", "--config=records/configs/fit_config.yaml"])
    udtube_python_interface(["predict", "--config=records/configs/fit_config.yaml"])