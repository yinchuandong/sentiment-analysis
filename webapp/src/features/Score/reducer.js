import { createReducer } from 'reduxsauce'
import * as Types from './actionType'

const INITIAL_STATE = {
  name: '',
  score: 0,
  processing: false
}

function handleChangeName(state = INITIAL_STATE, { payload }) {
  return {
    ...state,
    name: payload,
  }
}

function handleResetScore(state = INITIAL_STATE) {
  return {
    ...state,
    score: 0,
  }
}

function handleDoScoreRequest(state = INITIAL_STATE) {
  return {
    ...state,
    processing: true,
    score: 0,
  }
}

function handleDoScoreSuccess(state = INITIAL_STATE, { payload }) {
  console.log(payload)
  return {
    ...state,
    score: payload.score,
    processing: false
  }
}

function handleDoScoreFailure(state = INITIAL_STATE) {
  return {
    ...state,
    processing: false
  }
}
export const HANDLERS = {
  [Types.CHANGE_NAME]: handleChangeName,
  [Types.RESET_SCORE]: handleResetScore,
  [Types.DO_SCORE_REQUEST]: handleDoScoreRequest,
  [Types.DO_SCORE_SUCCESS]: handleDoScoreSuccess,
  [Types.DO_SCORE_FAILURE]: handleDoScoreFailure,
}

export default createReducer(INITIAL_STATE, HANDLERS)
