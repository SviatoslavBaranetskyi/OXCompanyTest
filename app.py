from datetime import datetime
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from database import db
from models import User, AnimeAnalysis
from config import Config

from utils import *
from ml_model import *
from validation import UserRegistrationValidator

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
jwt = JWTManager(app)

with app.app_context():
    db.create_all()


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    validator = UserRegistrationValidator(data)

    if not validator.validate():
        return jsonify({'message': validator.errors}), 400

    try:
        encrypted_email = encrypt(data['email'].encode(), Config.AES_KEY)

        new_user = User(username=data['username'], email=encrypted_email)
        new_user.set_password(data['password'])
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'message': 'User registered successfully.'}), 201

    except Exception as e:
        return jsonify({'message': f'Error occurred during registration: {str(e)}'}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    required_fields = ['username', 'password']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({'message': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    user = User.query.filter_by(username=data['username']).first()

    if user and user.check_password(data['password']):
        access_token = create_access_token(identity={'username': user.username, 'id': user.id})
        return jsonify(access_token=access_token), 200

    return jsonify({'message': 'Invalid username or password.'}), 401


@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_faces():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file.'}), 400

    try:
        image_path = save_user_image()
        full_image_path = os.path.join(Config.UPLOAD_FOLDER, image_path)

        predicted_label, confidence_score = predict(full_image_path, model, transform, label_map)

        if confidence_score < 0.5:
            raise ValueError(
                f"The identified cluster '{predicted_label}' might not be entirely accurate. Confidence is {confidence_score:.4f}.")

        response = {
            "predicted_label": predicted_label,
            "confidence_score": confidence_score,
            "image_path": f"/static/uploads/images/{image_path}",
            "message": "Prediction completed successfully."
        }

        user_info = get_jwt_identity()
        user_id = user_info['id']

        analysis = AnimeAnalysis(
            user_id=user_id,
            image_path=image_path,
            created_at=datetime.utcnow(),
            character_name=predicted_label
        )
        db.session.add(analysis)
        db.session.commit()
        db.session.refresh(analysis)

        return jsonify(response), 201

    except ValueError as e:
        return jsonify({'message': str(e)}), 400

    except Exception as e:
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500


@app.route('/analyses', methods=['GET'])
@jwt_required()
def get_user_analyses():
    try:
        user_info = get_jwt_identity()
        user_id = user_info['id']

        analyses = AnimeAnalysis.query.filter_by(user_id=user_id).all()

        analyses_list = [
            {
                "id": analysis.id,
                "image_path": f"/static/uploads/images/{analysis.image_path}",
                "created_at": analysis.created_at.isoformat(),
                "character_name": analysis.character_name
            }
            for analysis in analyses
        ]

        return jsonify({"analyses": analyses_list}), 200

    except Exception as e:
        return jsonify({'message': f'Error retrieving analyses: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
