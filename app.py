from flask import Flask,render_template,request
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50



# Load the VGG16 model
#model = VGG16(weights='none')
model = VGG16(weights=None)

# Uncomment and set weights_path if you want to load custom weights
# weights_path = 'path_to_your_weights.h5'
# model.load_weights(weights_path)

app = Flask(__name__)
model = ResNet50()


@app.route('/',methods=['GET'])
def render_index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def image_prediction():
    # Handle the POST request data here
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    imagefile = load_img(image_path,target_size=(224,224))
    imagefile = img_to_array(imagefile)
    imagefile = imagefile.reshape((1,imagefile.shape[0],imagefile.shape[1],imagefile.shape[2]))
    imagefile = preprocess_input(imagefile)
    yhat=model.predict(imagefile)
    label=decode_predictions(yhat)
    label=label[0][0]
    result='%s (%.2f%%)' % (label[1],label[2]*100)


    return render_template('index.html',prediction=result)


if __name__ == '__main__':
    app.run(port=3000, debug=True)