FROM mcr.microsoft.com/playwright/python:v1.57.0-noble

WORKDIR /app

COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
