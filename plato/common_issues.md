### Common Issues

This file contains resolutions for some common issues.

##### Environment Access Issues (More common in Virtual Environments)

1. Make sure "pip" path corresponds to the anaconda virtual environment or the default python path corresponding to the environment.

2. ``` “ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied”```

    while running: ``` pip install -r requirements.txt ```
    
    *Cause:* Access issue corresponding to system files/libraries vs virtual env. Add the “--user” flag:
    ``` pip install -r requirements.txt --user```

3. PyAudio Issues: If you would like to use speech un-comment "pyaudio" in "requirements.txt" file.
    PyAudio might cause errors while installing. You can do the following:
    - Through Anaconda (for any OS): 
        ``` conda install -c anaconda pyaudio ``` 
    - For MacOs:
        ``` 
        brew install portaudio
        pip install -r requirements.txt
        ``` 
    - For Ubuntu/Debian:
        ``` 
        sudo apt-get install python-pyaudio python3-pyaudio
        pip install -r requirements.txt
        ``` 
    - For Windows:
        ``` 
        pip install -r requirements.txt
        ``` 
        
    If installing on VM add a "--user" flag.
    
4. Google Text To Speech Library (Gtts):
    Common Error: ```AssertionError: gtts>=2.0.1 .dist-info directory not found```
    on Virtual machine:
        ``` 
        pip install setuptools --upgrade --user
        pip install --user --force-reinstall --ignore-installed --no-binary :all: gTTS
        ``` 
    or:
        ``` 
        pip install setuptools --upgrade
        pip install --force-reinstall --ignore-installed --no-binary :all: gTTS
        ``` 
        
5. Six:
    ``` 
    pip install six==1.11.0 --user
    ``` 





