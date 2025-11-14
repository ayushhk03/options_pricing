pipeline {
    agent any

    stages {
        
        stage('Checkout Code') {
            steps {
                git branch: 'master', url: 'https://github.com/ayushhk03/options_pricing.git'
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pip install pytest
                    pytest -v tests/
                '''
            }
        }

        stage('Package') {
            steps {
                sh '''
                    echo "Packaging project (no Docker needed)..."
                    mkdir -p build
                    cp -r src build/
                    cp -r run_experiment.py build/ || true
                    echo "Package created in ./build"
                '''
            }
        }

        stage('Deploy on Local Server') {
            steps {
                sh '''
                    echo "Simulating deployment..."
                    mkdir -p /Users/ayush/deployment
                    cp -r build/* /Users/ayush/deployment/
                    echo "Application deployed to /Users/ayush/deployment"
                '''
            }
        }
    }

    post {
        failure {
            echo "Pipeline failed. Check logs."
        }
        success {
            echo "Pipeline completed successfully!"
        }
    }
}
