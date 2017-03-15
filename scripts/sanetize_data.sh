#!/bin/bash

## skript for å rename bilder
v=$(basename "$PWD");
i=0;
for f in *;
    do
        mv "$f" "${v}_$i.jpg";
        let "i++";
    done;

## skript for å søke filer og slette filer med feil endelse
for f in *;
    do
        identify "$f" | grep -i GIF | awk '{print $1}' | xargs rm -f;
        identify "$f" | grep -i PNG | awk '{print $1}' | xargs rm -f;
    done;

for f in *;
    do
        identify "$f" | grep -i GIF;
        identity "$f" | grep -i PNG;
    done;