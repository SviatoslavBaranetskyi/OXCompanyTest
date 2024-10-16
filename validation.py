from models import User, AnimeAnalysis


class UserRegistrationValidator:
    def __init__(self, data):
        self.data = data
        self.errors = []

    def validate(self):
        self.check_required_fields()
        self.validate_email()
        self.validate_password()
        self.check_username_exists()

        return not self.errors

    def check_required_fields(self):
        required_fields = ['username', 'email', 'password']
        missing_fields = [field for field in required_fields if self.data.get(field) is None]
        if missing_fields:
            self.errors.append(f'Missing required fields: {", ".join(missing_fields)}')

    def validate_email(self):
        email = self.data.get('email')
        if email and not self.is_valid_email(email):
            self.errors.append('Invalid email address.')

    def validate_password(self):
        password = self.data.get('password')
        validations = [
            (not password, 'Password cannot be empty.'),
            (len(password) < 8, 'Password must be at least 8 characters long.'),
            (not any(char.isdigit() for char in password), 'Password must contain at least one digit.'),
            (not any(char.isalpha() for char in password), 'Password must contain at least one letter.'),
            (' ' in password, 'Password cannot contain spaces.')
        ]

        for condition, error_message in validations:
            if condition:
                self.errors.append(error_message)

    def check_username_exists(self):
        username = self.data.get('username')
        if username and User.query.filter_by(username=username).first():
            self.errors.append('Username already exists.')

    def is_valid_email(self, email):
        import re
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(email_regex, email) is not None

