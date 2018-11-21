FROM python:3.7
COPY requirements.txt /
RUN pip install -r requirements.txt
ADD News_Analysis.py /
COPY /* /
COPY Articles/* Articles/
COPY Output/* /Output/
COPY Data/* /Data/
COPY Result/* /Result/
CMD [ "python", "./News_Analysis.py" ]