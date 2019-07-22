# FromMotionToEmotion

## Wichtig

Zuerst müssen die Sensordaten heruntergeladen und entpackt werden.

Diese können hier https://github.com/mfreiwald/FromMotionToEmotion/releases/download/RAW/raw.zip heruntergeladen werden.

Der raw-Ordner muss im data-Ordner entpackt werden:

./data/raw/local

./data/raw/ned

./data/raw/positions


# Beschreibung der Daten

Beim Experiment gesammelte Sensordaten.

Es haben 21 Teilnehmer an der Studie teilgenommen.

Aufgrund von fehlerhaften Aufnahmen fehlen allerdings die Sensordaten für __P11__ und __P21__.


## Videos

Die betrachteten Videos:

### Entspannt

__World:__ Explore the World

Rundreise um den Globus - 3:17

https://www.youtube.com/watch?v=1_ifgJqLqTY

__Relax:__ Virtual Nature

Entspannende Orte - 4:43 (6:09)

https://www.youtube.com/watch?v=7AkbUfZjS5k


### Angespannt

__Nun:__ The Nun

Trailer zum Film The Nun - 3:28

https://www.youtube.com/watch?v=evzsN1BGR6A

__IT:__ IT

Trailer zum Film IT - 4:33 

https://www.youtube.com/watch?v=FHUErvVAeIw

## Sensordaten

Jede Datei gibt den Teilnehmer und das betrachtete Video an (z.B. P01_World)

### ned

Vektoren aus dem NED (North-East-Down) Koordinatensystem

Erste Spalte: Timestamp

Aufbau der weiteren Spalten: Sensor_heading|alignment|orientation_q|v_x|y|z|[w]

z.B. Head_orientation_q_x

Sensor: Name des Sensors

Heading: ...

Alignment: ...

Orientation: Orientierungsvektor des Sensors

q: Darstellung als Quaternion (x, y, z, w)

v: Darstellung als Rotationsvektor (x, y, z) in Grad (0 - 360)


### local

Vektoren aus dem lokalen Unity-Koordinatensystem

Aufbau siehe oben


### positions

Position der Sensoren aus dem 3D-Koordinatensystem von Unity

Erste Spalte: Timestamp

Aufbau der weiteren Spalten: 

Sensor_position_v_x|y|z

Sensor_rotation_q|v_x|y|z|[w]

Sensor: Name des Sensors

position: 3D-Koordinate (Position) des Sensors

rotation: Drehung des Sensors, dargestellt als Quaternion und Rotationsvektor


## Metadaten

### videos.csv

Selbstbeurteilung der Teilnehmer nach jedem Video

1 - sehr entspannt

2 - entspannt

3 - neutral

4 - angespannt

5 - sehr angespannt


### questionnaire.csv

Auswertung der Fragebögen:

Gibt die Abspielreihenfolge der Videos an.

Spalten:

__beforeFeeling:__ Wie fühlst du dich gerade?

1 - sehr schlecht

2 - schlecht

3 - geht so

4 - gut

5 - sehr gut


__beforeRanking:__ Wie angespannt bist du gerade?

1 - sehr entspannt

2 - entspannt

3 - neutral

4 - angespannt

5 - sehr angespannt


__fearful:__ Würdest du dich als ängstlich bezeichnen?

1 - nicht ängstlich

2 - kaum ängstlich

3 - ein wenig ängstlich

4 - ängstlich

5 - sehr ängstlich


__afterFeeling:__ Wie fühlst du dich nach den Videos?

1 - sehr schlecht

2 - schlecht

3 - geht so

4 - gut

5 - sehr gut 



