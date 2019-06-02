import { bindActionCreators } from 'redux'
import { connect } from 'react-redux'
import { withRouter } from 'react-router'
import ScoreForm from './view'
import Actions from './action'
import { getName, getScore, getProcessing } from './selector'


const mapStateToProps = (state, props) => ({
  name: getName(state, props),
  score: getScore(state, props),
  processing: getProcessing(state, props)
})

const mapDispatchToProps = dispatch => ({
  actions: bindActionCreators(Actions, dispatch),
})

export default withRouter(connect(
  mapStateToProps,
  mapDispatchToProps,
)(ScoreForm))
