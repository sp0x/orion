node {
    def app

    stage('Clone repository') {
        /**/
        checkout scm
    }

    stage('Build image') {
       /* This builds the docker image*/
            newImage = docker.build("orion")
            docker.withRegistry("https://344965022394.dkr.ecr.us-east-2.amazonaws.com", "ecr:us-east-2:aws_ecr") {
                newImage.push()
            }
    } 
    stage('Deploy') {
        sh 'curl --request POST "https://deploy.netlyt.io/?token=TOKEN&hook=orion" > /dev/null 2>&1 &'
    }
    stage('Notify') {
        slackSend baseUrl: 'https://netlyt.slack.com/services/hooks/jenkins-ci/', channel: 'dev', color: 'good',
         message: "${env.JOB_NAME} - #${env.BUILD_NUMBER} Successfull (<${env.BUILD_URL}|Open>)",
          teamDomain: 'netlyt',
           tokenCredentialId: 'jenkins-slack-integration'
    }
 
}