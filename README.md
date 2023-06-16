# SageMaker-Docker-Local

```

# Start container

docker run \
-v /home/ec2-user/SageMaker:/opt/ml/model \
--cpu-shares 512 \
-p 8080:8080 \
763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.0-cu117 \
serve

# Sample request

curl -X POST http://localhost:8080/invocations -H "Content-type: text/plain" "This is a sample test string"

```

# [Blog](https://towardsdatascience.com/debugging-sagemaker-endpoints-with-docker-7a703fae3a26)
