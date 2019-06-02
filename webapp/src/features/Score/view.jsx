import React, { Component } from "react";
import PropTypes from "prop-types";
import { Form, Spin, Input, Button } from "antd";

import "./view.less";

class Score extends Component {
  static defaultProps = {
    name: "",
    score: 0
  };

  static contextTypes = {
    router: PropTypes.object.isRequired
  };

  componentDidUpdate = () => {

  };

  componentDidMount() {

  }

  handleSubmit = e => {
    //do score whenever click the button
    const { name, actions } = this.props;
    e.preventDefault();
    this.props.form.validateFields((err, values) => {
      if (!err && name !== values.name) {
        actions.changeName(values.name);
        actions.doScoreRequest(values.name);
      }
    });
  };

  scoreDetail = (name, score) => {
    if (name === "") {
      return "Score yout text!";
    } else {
      return (
        <span className="score-detail">
          <span>your score is: <b>{score}</b></span>
        </span>
      );
    }
  };

  render() {
    const { getFieldDecorator } = this.props.form;
    const { name, score, processing } = this.props;
    return (
      <div className={name === "" ? "score-form" : "score-form up"}>
        <Spin size="large" spinning={processing} />
        <Form onSubmit={this.handleSubmit}>
          <Form.Item className="score-in">
            {getFieldDecorator("name", {
              rules: [
                { required: true, message: "Please input your comments to test!" }
              ]
            })(<Input placeholder="score text" />)}
          </Form.Item>
          <Form.Item className="score-button">
            {this.scoreDetail(name, score)}
            <Button type="primary" htmlType="submit" className="btn-score">
              Score
            </Button>
          </Form.Item>
        </Form>
      </div>
    );
  }
}

const ScoreForm = Form.create({
  name: "normal_score",
  //keep the value in the input area the same as the url's hash 保持输入框里面的值跟url链接上的参数一致
  mapPropsToFields(props) {
    return {
      name: Form.createFormField({
        ...props.name,
        value: props.name
      })
    };
  }
})(Score);

Score.propTypes = {
  name: PropTypes.string,
  score: PropTypes.number
};

export default ScoreForm;
