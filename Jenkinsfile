pipeline {
    agent any

    environment {
        VENV = "venv"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/ayushhk03/options_pricing.git'
            }
        }

        stage('Build / Compile / Package') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt

                    # Install your project package (since you have setup.py)
                    pip install -e .
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest -v tests/
                '''
            }
        }

        stage('Docker Build (Packaging)') {
            steps {
                sh '''
                    docker build -t options_pricing_app:latest .
                '''
            }
        }

        stage('Deploy on Local Server') {
            steps {
                sh '''
                    docker stop options_pricing_app || true
                    docker rm options_pricing_app || true

                    docker run -d --name options_pricing_app -p 8000:8000 options_pricing_app:latest
                '''
            }
        }
    }

    post {
        success { echo "Pipeline completed successfully!" }
        failure { echo "Pipeline failed. Check the logs." }
    }
}
