import http from '../utils/HttpUtil'

const predict = data => {
  console.log(data)
  return http.post('/textcnn/predict', data)
}

export default { predict }
