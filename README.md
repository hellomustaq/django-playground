# API/ChatbotV4_DjangoRest/

It is the current best usable version till now (October 28,2019)
It uses Keras Seuqential Deep Learning Model. Being an AI model, its responses cannot be always 100% correct, but it responds much better that previous versions. 
Their are some critical dependency issues with this version. 
The dependencies, that work fine with it are as follows:  
[packages]  
django = "*"  
djangorestframework = "*"  
nltk = "*"  
numpy = "*"  
tensorflow = "1.14.0"  
keras = "2.2.4"  
pandas = "*"  

The tensorflow and keras versions need to be exactly as listed above.

To run this program, the 'intents.json' file and an empty 'intentsPreviousCopy.json' file need to be inside the appropriate directory (this root directory along with 'manage.py' file).   
Then running the command 'python manage.py runserver' will start the program.   
It was tested on Postman by going to the url: 'http://127.0.0.1:8000/chatbotV4/'.  
Inisde Postman body, the query strings need to be enclosed withing double quotes like "What is your name?"

