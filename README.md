# 3 Faces Detection

1. [Cargar el archivo Haar_face.xml ](#schema1)
2. [Crear faces_rect](#schema2)
3. [Obtener el número de caras](#schema3)
4. [Detectar las caras con un rectángulo verde](#schema4)
5. [Face Recognition with OpenCV's built-in recognizer](#schema5)

<hr>

<a name="schema1"></a>

# 1. Cargar el archivo haar_facel-xml
~~~python
haar_cascade = cv.CascadeClassifier('haar_face.xml')
~~~

<hr>

<a name="schema2"></a>

# 2. Crear faces_rect

~~~pytho
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
~~~
<hr>

<a name="schema3"></a>

# 3. Obtener el número de caras

~~~python
print(f'Number of faces found = {len(faces_rect)}')
~~~
![faces](./images/001.png)

<hr>

<a name="schema4"></a>

# 4. Detectar las caras con un rectángulo verde

~~~python
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
~~~
![faces](./images/002.png)
![faces](./images/003.png)

Modificando el `minNeighbors` obtenemos mas caras
~~~python
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
~~~
![faces](./images/004.png)


<hr>

<a name="schema5"></a>

# 5. Face Recognition with OpenCV's built-in recognizer