# SSD-Keras

Refactory del progetto 
https://github.com/pierluigiferrari/ssd_keras

Il file `configSSD.json` contiene la configurazione della rete, delle classi, del training, i dataset, eccetera.

E' possibile indicare quale tipologia di modello utilizzare (`300` oppure `512`)

Per eseguire il training, lanciare il file `training.py`.

Per eseguire il test, lanciare il file `test.py`.

Il file `show_features.py` serve a visualizzare le immagini elaborate da ogni filtro convoluzionale.

Nella directory `logs` vengono creati i log per **TensorBoard**.