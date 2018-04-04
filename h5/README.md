I pesi di **BASE** sono scaricabili da qui: 

Versione 300 (da rinominare in `base300.h5`): 
https://drive.google.com/open?id=17G1J4zEpFwiOzgBmq886ci4P3YaIz8bY

Versione 512 (da rinominare in `base512.h5`):
https://drive.google.com/open?id=1wGc368WyXSHZOv4iow2tri9LnB0vm9X-


Il file `pretrained.h5` se esiste è quello che viene caricato all'inizio del training come partenza (è impostato sulle classi indicate in `configSSD.json`).

Il file `result.h5` se esiste è quello creato alla fine dell'allenamento.

Il file `chkpnt_best.h5` se esiste è quello creato durante l'allenamento come checkpoint migliore. Può essere rinominato in `pretrained.h5` se si interrompe l'allenamento e successivamente si vuole ripartire da quel punto.

Tutti i nomi ed i path dei file sono comunque configurabili da `configSSD.json`.


