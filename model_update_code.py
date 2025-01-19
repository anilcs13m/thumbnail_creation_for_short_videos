from tensorflow.keras.models import load_model

# Load the model without compiling
model = load_model('face_validation_model.h5', compile=False)

# Manually set the optimizer with the updated argument
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the updated model
model.save('updated_face_validation_model.h5')


