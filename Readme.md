##Inhaltszusammenfassung

Dieses Repo basiert auf dem [ASCAD](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)-Repo. Die ursprüngliche Readme ist unter Readme_ASCAD zu finden. Hier findet sich nur eine rudimentäre Zusammenfasuung des hinzugefügten Inhalts.

#crosscal.py

In crossval.py wird die Crossvalidierung der Ergebnisse für die Klassifizierung nach Labelidentität und Hamming-Gewichts-Modell ohne Data Augementation vorgenommen. Für die Änderung der Stategie muss die Variable hamming verändert werden. FÜr die asuführung des codes ist es nötig das die gewünschten Ablageordner für die Erebnisse bereit bestehen. Außedem müssen die ASCAD-Daten gemäß den Vorgaben in Readme_ASCAD vorgenommen werden.

#data_addition.py

Ähnlich wie crossval.py bloß das hier Data Augementation vorgenommen wird. Um zwichen SMOTE und Erwartungswertmodellierung zu wechseln muss die jeweilge andere Methode auskomentiert werden.

Die Darstellung der traces und der Datenmangen sind ebenfalls hier Implementiert. Sie sind jedoch auskommentiert um nicht die Effizienz der Berechnungen zu behindern.

#testtest.py

Setzt die reduktion der Daten um. Dieser Anstz wird in der Arbeit verworfen, weshalb keine Crossvalidierung implementiert ist

#tracehelper1.py
Führt die Berechnungen zum SNR und den Erwartungswertdiagrammen durch


#accum_means.py
accum_means.py setzt die Ergebnisse der Verschiedenen Crossvalidationen zusamen in ein Diagramm 

#hammingDiagramm.py
hammingDiagramm.py visualisiert die Verteilung der Label in ASCAD