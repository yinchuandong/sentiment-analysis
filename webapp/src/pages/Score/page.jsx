import React, { Component } from "react";
import Score from "../../features/Score";
import "./index.less";

class ScorePage extends Component {
  render() {
    return (
      <div className="score-wrap">
        <section className="score-main">
          <Score hash={this.props.location.hash} />
        </section>
      </div>
    );
  }
}

export default ScorePage;
